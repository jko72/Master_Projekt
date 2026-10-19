"""CPU checks for the SemanticKITTI remission MOS variants."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mos_dataset import MOSFrameDataset, TemporalMOSDataset
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from mos_models import build_mos_model
from helper.mos_input import channel_count
from evaluate_mos_benchmark import temporal_input, single_frame_input, load_adapter
from train_temporal_mos_e2e import read_config
from train_mos import _check_pretrain_channels
from train_mos import train_one_epoch


def fixture_scan(tmp_path, n_frames=2):
    seq_dir = tmp_path / "00"
    scans = seq_dir / "velodyne"
    labels = seq_dir / "mos_labels"
    scans.mkdir(parents=True)
    labels.mkdir()
    # Two points collide at the same pixel. The nearer one must supply XYZI.
    frames = [np.array([[10, 0, 0, .9], [5, 0, 0, .4], [0, 5, 0, .8],
                        [-3, 0, 0, np.nan]], dtype=np.float32)] * n_frames
    paths = []
    for i, points in enumerate(frames):
        path = scans / f"{i:06d}.bin"
        points.tofile(path)
        np.save(labels / f"{i:06d}.npy", np.zeros((32, 64), dtype=np.int16))
        paths.append(str(path))
        for offset in (1, 2):
            folder = seq_dir / f"residual_images_{offset}"
            folder.mkdir(exist_ok=True)
            np.save(folder / f"{i:06d}.npy", np.full((32, 64), offset, dtype=np.float32))
    pose0 = np.eye(4)
    pose1 = np.eye(4)
    pose0[0, 3] = 1
    seq = {"seq_id": "00", "paths": paths, "poses": [pose0] + [pose1] * (n_frames - 1)}
    cfg = {"model_params": {"grid_height": 32, "grid_width": 64, "FOV_UP": 3,
                             "FOV_DOWN": -25, "min_range": 0.1, "max_range": 80},
           "mos_data_params": {"input_mode": "range_xyz_intensity", "normalization": "intensity_only",
                               "previous_offsets": [1], "scan_rate_hz": 10,
                               "residual_offsets": [1, 2]}}
    return seq, cfg


def check_projection_order_collision_normalization_and_legacy(tmp_path):
    seq, cfg = fixture_scan(tmp_path)
    ds = MOSFrameDataset([seq], cfg, input_mode="range_xyz_intensity", residual_offsets=[])
    x, _, meta = ds[0]
    assert x.shape == (5, 32, 64)
    assert meta["channel_names"] == ["x", "y", "z", "range", "intensity"]
    selected = (x[0] == 5) & (x[3] == 5)
    assert selected.sum() == 1
    assert torch.allclose(x[4, selected], torch.tensor([(.4 - .21) / .16]), atol=1e-6)
    assert torch.isfinite(x).all()
    assert torch.all(x[:, x[3] == 0] == 0)
    cfg["mos_data_params"]["normalization"] = "none"
    raw = MOSFrameDataset([seq], cfg, input_mode="range_xyz_intensity", residual_offsets=[])[0][0]
    legacy = MOSFrameDataset([seq], cfg, input_mode="range_xyz", residual_offsets=[])[0][0]
    assert torch.equal(raw[:4, raw[3] > 0], legacy[:, raw[3] > 0])
    assert torch.count_nonzero(legacy[3]) == torch.count_nonzero(raw[3]) + 1
    assert torch.equal(raw[4, selected], torch.tensor([.4]))
    residual = MOSFrameDataset([seq], cfg, input_mode="range_xyz_intensity_residual",
                               residual_offsets=[1, 2])[0][0]
    assert residual.shape == (7, 32, 64)
    assert torch.equal(residual[:5], raw)
    assert torch.all(residual[5] == 1) and torch.all(residual[6] == 2)
    assert channel_count("range_xyz_intensity_residual", range(5)) == 10


def check_paper_normalization_and_benchmark_input(tmp_path):
    seq, cfg = fixture_scan(tmp_path)
    cfg["mos_data_params"]["normalization"] = "paper_normalization"
    x = MOSFrameDataset([seq], cfg, input_mode="range_xyz_intensity", residual_offsets=[])[0][0]
    selected = x[3] != 0
    assert torch.allclose(x[0, selected & (x[4] < 2)],
                          (torch.tensor([5.]) - 10.88) / 11.47, atol=1e-5)
    assert torch.all(x[:, ~selected] == 0)
    bench = single_frame_input(Path(seq["paths"][0]), cfg,
                               Path(seq["paths"][0]).parents[1], "000000")[0]
    assert torch.allclose(bench, x)


def check_temporal_intensity_survives_pose_and_benchmark(tmp_path):
    seq, cfg = fixture_scan(tmp_path)
    ds = TemporalMOSE2EDataset([seq], cfg)
    x, valid, dt, _, meta = ds[0]
    assert x.shape == (2, 5, 32, 64)
    assert valid.shape == (2, 32, 64)
    assert np.allclose(dt.numpy(), [-.1, 0])
    assert meta["channel_names"][-1] == "intensity"
    assert torch.isfinite(x).all()
    past_point = (x[0, 0] == 6) & valid[0]
    assert past_point.sum() == 1
    assert torch.allclose(x[0, 4, past_point], torch.tensor([(.4 - .21) / .16]), atol=1e-6)
    scans = [Path(p) for p in seq["paths"]]
    inputs, _ = temporal_input(scans, 1, seq["poses"], cfg, "strict", True)
    assert torch.allclose(inputs[0][0], x)
    cfg["mos_data_params"]["previous_offset"] = 1
    siamese = TemporalMOSDataset([seq], cfg, input_mode="range_xyz_intensity", residual_offsets=[])
    siamese_x, _, siamese_meta = siamese[0]
    assert siamese_x.shape == (2, 5, 32, 64)
    assert siamese_meta["channel_names"][-1] == "intensity"
    siamese_inputs, _ = temporal_input(scans, 1, seq["poses"], cfg, "strict", False)
    assert torch.allclose(siamese_inputs[0][0], siamese_x)


def check_new_model_forwards_and_config_validation():
    torch.set_num_threads(1)
    for name in ("salsanext_mos", "salsanext_temporal_mos", "salsanext_temporal_e2e"):
        cfg = {"mos_model_params": {"name": name, "in_channels": 5}}
        model = build_mos_model(cfg).eval()
        with torch.no_grad():
            if name == "salsanext_mos":
                y = model(torch.zeros(1, 5, 32, 64))
            elif name == "salsanext_temporal_mos":
                y = model(torch.zeros(1, 2, 5, 32, 64))
            else:
                y = model(torch.zeros(1, 2, 5, 32, 64),
                          torch.ones(1, 2, 32, 64, dtype=torch.bool),
                          torch.tensor([[-.1, 0.]]))
        assert y.shape == (1, 2, 32, 64)
    path, cfg = read_config(ROOT / "src/configs/mos_semanticKitti_temporal_e2e_intensity.yaml")
    assert path.is_file() and cfg["mos_model_params"]["in_channels"] == 5
    siamese_cfg = yaml.safe_load((ROOT / "src/configs/mos_semanticKitti_siamese_intensity.yaml").read_text())
    assert siamese_cfg["mos_data_params"]["previous_offset"] == 3
    assert siamese_cfg["mos_model_params"]["name"] == "salsanext_temporal_mos"


def check_benchmark_checkpoint_adapter(tmp_path):
    cfg = {"mos_model_params": {"name": "salsanext_mos", "in_channels": 5},
           "mos_data_params": {"input_mode": "range_xyz_intensity", "normalization": "intensity_only",
                               "residual_offsets": []}}
    model = build_mos_model(cfg)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, path)
    loaded, loaded_cfg, _ = load_adapter(path, torch.device("cpu"))
    assert loaded.in_channels == 5 and loaded_cfg == cfg
    cfg["mos_model_params"]["in_channels"] = 4
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, path)
    with unittest.TestCase().assertRaisesRegex(ValueError, "requires 5 channels"):
        load_adapter(path, torch.device("cpu"))


def check_pretrain_order_guard():
    target = {"mos_model_params": {"in_channels": 5},
              "mos_data_params": {"input_mode": "range_xyz_intensity"}}
    four = {"encoder_state_dict": {"down_cntx.conv1.weight": torch.zeros(32, 4, 3, 3)}}
    five = {"encoder_state_dict": {"down_cntx.conv1.weight": torch.zeros(32, 5, 3, 3)}}
    with unittest.TestCase().assertRaisesRegex(ValueError, "Set pretrained_encoder_adapt_input_channels"):
        _check_pretrain_channels(four, target, False)
    _check_pretrain_channels(four, target, True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "order is unverified"):
        _check_pretrain_channels(five, target, False)
    five["config"] = {"model_params": {"name": "salsanext_temporal_mae", "grid_channels": 5},
                      "pretrain_params": {"task": "tmae_rangeview"}}
    _check_pretrain_channels(five, target, False)


def check_siamese_cpu_train_step(tmp_path):
    torch.set_num_threads(1)
    seq, cfg = fixture_scan(tmp_path)
    cfg["mos_data_params"]["previous_offset"] = 1
    cfg["mos_model_params"] = {"name": "salsanext_temporal_mos", "in_channels": 5}
    cfg["mos_train_params"] = {"clip_grad_norm": 5.0}
    ds = TemporalMOSDataset([seq], cfg, input_mode="range_xyz_intensity", residual_offsets=[])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
    model = build_mos_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    before = model.encoder.down_cntx.conv1.weight.detach().clone()
    loss, counts, _ = train_one_epoch(model, loader, torch.nn.CrossEntropyLoss(ignore_index=-1),
                                      optimizer, "cpu", cfg)
    assert np.isfinite(loss) and counts["valid_pixels"] > 0
    assert not torch.equal(before, model.encoder.down_cntx.conv1.weight)


def check_current_e2e_offsets_and_cpu_step(tmp_path):
    torch.set_num_threads(1)
    seq, cfg = fixture_scan(tmp_path, n_frames=6)
    cfg["mos_data_params"]["previous_offsets"] = [5, 2]
    cfg["mos_model_params"] = {"name": "salsanext_temporal_e2e", "in_channels": 5}
    ds = TemporalMOSE2EDataset([seq], cfg)
    x, valid, dt, y, _ = ds[0]
    assert x.shape == (3, 5, 32, 64)
    assert np.allclose(dt.numpy(), [-0.5, -0.2, 0.0])
    model = build_mos_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x.unsqueeze(0), valid.unsqueeze(0), dt.unsqueeze(0))
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(logits, y.unsqueeze(0))
    assert torch.isfinite(loss)
    loss.backward()
    for part in (model.encoder, model.temporal, model.decoder):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in part.parameters())
    optimizer.step()


class MOSIntensityTests(unittest.TestCase):
    def _with_scan(self, check):
        with tempfile.TemporaryDirectory() as directory:
            check(Path(directory))

    def test_projection_order_collision_normalization_and_legacy(self):
        self._with_scan(check_projection_order_collision_normalization_and_legacy)

    def test_paper_normalization_and_benchmark_input(self):
        self._with_scan(check_paper_normalization_and_benchmark_input)

    def test_temporal_intensity_survives_pose_and_benchmark(self):
        self._with_scan(check_temporal_intensity_survives_pose_and_benchmark)

    def test_new_model_forwards_and_config_validation(self):
        check_new_model_forwards_and_config_validation()

    def test_benchmark_checkpoint_adapter(self):
        self._with_scan(check_benchmark_checkpoint_adapter)

    def test_pretrain_order_guard(self):
        check_pretrain_order_guard()

    def test_siamese_cpu_train_step(self):
        self._with_scan(check_siamese_cpu_train_step)

    def test_current_e2e_offsets_and_cpu_step(self):
        self._with_scan(check_current_e2e_offsets_and_cpu_step)
