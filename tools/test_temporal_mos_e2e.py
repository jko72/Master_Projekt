#!/usr/bin/env python3
"""Small CPU invariants for the supervised temporal MOS path."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from models.range_view_local_temporal_attention import LocalHistoricalAttention, local_candidates
from mos_models import build_mos_model
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from train_temporal_mos_e2e import amp_settings, read_config
from evaluate_temporal_mos_e2e import apply_history_mode


class TemporalE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.cfg = read_config("src/configs/mos_semanticKitti_temporal_e2e.yaml")
        torch.set_num_threads(2)

    def test_cylindrical_candidates_and_mask(self):
        grid = torch.arange(12).reshape(1, 1, 1, 3, 4).float()
        candidates = local_candidates(grid, 1).reshape(1, 3, 4, 9)
        self.assertEqual(candidates[0, 1, 0, 3].item(), 7.0)  # left wraps to col 3
        self.assertEqual(candidates[0, 0, 0, 1].item(), 0.0)  # top has no wrap
        attention = LocalHistoricalAttention(8, heads=2, radius=1, compare_channels=4)
        current = torch.randn(1, 8, 3, 4)
        history = torch.randn(1, 1, 8, 3, 4)
        valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)
        absent = torch.zeros(1, 1, 1, 3, 4, dtype=torch.bool)
        key_inputs = []
        value_inputs = []
        key_hook = attention.key.register_forward_pre_hook(lambda _module, args: key_inputs.append(args[0].detach().clone()))
        value_hook = attention.value.register_forward_pre_hook(lambda _module, args: value_inputs.append(args[0].detach().clone()))
        fused, diag = attention(current, history, valid, absent, torch.tensor([[-0.3]]), True)
        key_hook.remove(); value_hook.remove()
        self.assertTrue(torch.equal(key_inputs[0], history[:, 0]))
        self.assertTrue(torch.equal(value_inputs[0], history[:, 0]))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.allclose(diag["null_weight"], torch.ones_like(diag["null_weight"])))
        self.assertTrue(torch.all(diag["best_match"] == 0))
        # No historical keys means changing history cannot change the output.
        fused2 = attention(current, history + 100, valid, absent, torch.tensor([[-0.3]]))
        self.assertTrue(torch.allclose(fused, fused2, atol=1e-6))

    def test_model_t2_t3_backward_checkpoint(self):
        for t in (2, 3):
            model = build_mos_model(self.cfg)
            x = torch.randn(1, t, 4, 16, 32)
            valid = torch.ones(1, t, 16, 32, dtype=torch.bool)
            valid[:, :-1, :2] = False
            dt = torch.tensor([[-0.1 * (t - i - 1) for i in range(t)]])
            y = torch.randint(0, 2, (1, 16, 32))
            y[:, :2] = -1
            logits = model(x, valid, dt)
            loss = F.cross_entropy(logits, y, ignore_index=-1)
            loss.backward()
            self.assertEqual(logits.shape, (1, 2, 16, 32))
            self.assertTrue(torch.isfinite(loss))
            for module in (model.encoder, model.temporal, model.decoder):
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.abs().sum() > 0 for p in module.parameters()))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "checkpoint.pt"
                torch.save({"model_state_dict": model.state_dict()}, path)
                clone = build_mos_model(self.cfg)
                clone.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"], strict=True)
                model.eval(); clone.eval()
                with torch.no_grad():
                    self.assertTrue(torch.allclose(model(x, valid, dt), clone(x, valid, dt), atol=1e-6))

    def test_history_modes_touch_only_past(self):
        x = torch.randn(1, 3, 4, 2, 2)
        valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
        zero, zero_valid = apply_history_mode(x, valid, "zero")
        copy, copy_valid = apply_history_mode(x, valid, "copy_current")
        self.assertTrue(torch.equal(zero[:, -1], x[:, -1]))
        self.assertTrue(torch.equal(copy[:, -1], x[:, -1]))
        self.assertTrue(torch.all(zero[:, :-1] == 0))
        self.assertTrue(torch.all(~zero_valid[:, :-1]))
        self.assertTrue(torch.equal(copy[:, 0], x[:, -1]))
        self.assertTrue(torch.equal(copy[:, 1], x[:, -1]))
        self.assertTrue(torch.equal(copy_valid[:, :-1], valid[:, :-1]))

    def test_dataset_offsets_projection_and_current_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq_dir = Path(tmp) / "00"
            (seq_dir / "velodyne").mkdir(parents=True)
            (seq_dir / "mos_labels").mkdir()
            paths, poses = [], []
            for i in range(4):
                path = seq_dir / "velodyne" / f"{i:06d}.bin"
                np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float32).tofile(path)
                np.save(seq_dir / "mos_labels" / f"{i:06d}.npy", np.ones((16, 32), dtype=np.int16))
                paths.append((str(path), ""))
                pose = np.eye(4)
                pose[0, 3] = i * 0.1
                poses.append(pose)
            cfg = {**self.cfg, "model_params": {**self.cfg["model_params"], "grid_height": 16, "grid_width": 32},
                   "mos_data_params": {**self.cfg["mos_data_params"], "previous_offsets": [2, 1]}}
            dataset = TemporalMOSE2EDataset([{"seq_id": "00", "paths": paths, "poses": poses}], cfg)
            self.assertEqual(len(dataset), 2)
            x, valid, dt, y, meta = dataset[0]
            self.assertEqual(x.shape, (3, 4, 16, 32))
            self.assertEqual(valid.shape, (3, 16, 32))
            self.assertTrue(torch.equal(dt, torch.tensor([-0.2, -0.1, 0.0])))
            self.assertEqual(meta["frame_index"], 2)
            self.assertEqual([Path(p).stem for p in meta["history_paths"]], ["000000", "000001"])
            self.assertEqual((y != -1).sum().item(), valid[-1].sum().item())
            self.assertTrue(torch.all(y[~valid[-1]] == -1))

    def test_existing_builder_entries(self):
        for name in ("salsanext_mos", "salsanext_temporal_mos"):
            cfg = {**self.cfg, "mos_model_params": {**self.cfg["mos_model_params"], "name": name}}
            self.assertIsInstance(build_mos_model(cfg), torch.nn.Module)

    def test_amp_default(self):
        enabled, dtype = amp_settings(self.cfg, torch.device("cpu"))
        self.assertFalse(enabled)
        self.assertEqual(dtype, torch.bfloat16)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            enabled, dtype = amp_settings(self.cfg, torch.device("cuda"))
            self.assertTrue(enabled)
            self.assertEqual(dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
