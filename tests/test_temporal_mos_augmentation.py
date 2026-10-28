"""Real projection, worker/DDP, and training checks for train-only augmentation."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from helper.mos_augmentation import augmentation_settings, TemporalMOSAugmentation
from helper.mos_input import normalization, normalize_feature_tensor
from helper.residual_helper import transform_points
from temporal_mae_dataset import project_range_features
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
import train_temporal_mos_e2e as trainer
import evaluate_temporal_mos_e2e as evaluator
from evaluate_mos_benchmark import temporal_input

REFERENCE = ROOT / 'src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz_w290.yaml'
AUGMENTED = REFERENCE.with_name(REFERENCE.stem + '_aug.yaml')


def fixture(directory, height=16, width=64):
    directory = Path(directory) / '00'
    (directory / 'velodyne').mkdir(parents=True, exist_ok=True)
    (directory / 'mos_labels').mkdir(exist_ok=True)
    # Populate distinct angular cells and add a nearer collision, plus a NaN remission.
    phi = np.linspace(-np.pi + .02, np.pi - .02, width * 2)
    points = np.stack([10 * np.cos(phi), 10 * np.sin(phi),
                       np.full_like(phi, -.5), np.full_like(phi, .5)], axis=1).astype(np.float32)
    points = np.concatenate([points, [[5, 0, -.25, .3], [0, -4, 0, np.nan]]]).astype(np.float32)
    paths, poses = [], []
    for index in range(6):
        path = directory / 'velodyne' / f'{index:06d}.bin'
        points.tofile(path)
        np.save(directory / 'mos_labels' / f'{index:06d}.npy',
                np.indices((height, width)).sum(0).astype(np.int16) % 2)
        pose = np.eye(4)
        pose[0, 3] = .1 * index
        paths.append(str(path))
        poses.append(pose)
    cfg = yaml.safe_load(REFERENCE.read_text())
    cfg['model_params'].update(grid_height=height, grid_width=width)
    cfg['mos_data_params'].update(previous_offsets=[2, 1], mos_label_folder='mos_labels',
                                  normalization='none', train_sequences=['00'])
    cfg['mos_log_params']['use_tensorboard'] = False
    return dict(seq_id='00', paths=paths, poses=poses), cfg


def enabled(*names):
    return {'enabled': True, **{name: {'enabled': True, 'probability': 1.0} for name in names}}


NAMES = ('azimuth_rotation', 'historical_point_dropout', 'intensity_jitter')


def legacy_sample(ds, index=0):
    """Independent reference of the original dataset path, without augmentation calls."""
    sample = ds.samples[index]
    seq = ds.sequence_map[sample['seq_id']]
    i = sample['frame_index']
    scans = [transform_points(ds._read_points(seq['paths'][i-off]),
                             np.linalg.solve(ds._pose(seq, i), ds._pose(seq, i-off)))
             for off in ds.offsets] + [ds._read_points(sample['scan_path'])]
    features, masks = zip(*(project_range_features(p, ds.H, ds.W, ds.theta_range,
                            ds.min_range, ds.max_range, include_intensity=True,
                            explicit_occupancy=True) for p in scans))
    x = torch.stack([normalize_feature_tensor(f, m, normalization(ds.cfg, ds.temporal_input_mode))
                     for f, m in zip(features, masks)])
    valid = torch.stack(masks)[:, 0].bool()
    y = torch.from_numpy(ds._load_mos_label(sample['mos_label_path']).astype(np.int64)).masked_fill(~valid[-1], -1)
    dt = torch.tensor([-off/ds.scan_rate_hz for off in ds.offsets] + [0.])
    return x, valid, dt, y


def distributed_augmentation_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'),
                            rank=rank, world_size=2)
    try:
        seq = np.load(Path(directory) / 'seq.npy', allow_pickle=True).item()
        cfg = yaml.safe_load((Path(directory) / 'cfg.yaml').read_text())
        def load():
            trainer.seed_all(42 + rank)
            with patch.object(trainer, 'make_sequences', return_value=[seq]):
                loader = trainer.make_loader(cfg, 'train', 1, 2, torch.device('cpu'))
                loader.sampler.set_epoch(0)
                return next(iter(loader))
        first, again = load(), load()
        for a, b in zip(first[:4], again[:4]):
            assert torch.equal(a, b)
        # Ranks have distinct worker RNG streams; test these on the SAME sample too.
        trainer.seed_all(42 + rank)
        sample = TemporalMOSE2EDataset([seq], cfg)[0][0]
        gathered = [torch.empty_like(sample) for _ in range(2)]
        dist.all_gather(gathered, sample)
        assert not torch.equal(*gathered)
        model = trainer.wrap_distributed(trainer.build_mos_model(cfg), torch.device('cpu'))
        x, valid, dt, y = first[:4]
        loss = trainer.make_criterion(cfg, torch.device('cpu'))(model(x, valid, dt), y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        dist.destroy_process_group()


class TemporalAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mos-augmentation-')
        self.addCleanup(self.tmp.cleanup)
        self.seq, self.cfg = fixture(self.tmp.name)

    def dataset(self, settings=None, split='train'):
        cfg = copy.deepcopy(self.cfg)
        if settings is not None:
            cfg['mos_augmentation'] = settings
        return TemporalMOSE2EDataset([self.seq], cfg, split=split)

    def assert_tensors_equal(self, a, b):
        for left, right in zip(a[:4], b[:4]):
            self.assertTrue(torch.equal(left, right))

    def test_disabled_and_nontraining_match_original_without_rng_draws(self):
        for norm in ('none', 'intensity_only', 'paper_normalization'):
            self.cfg['mos_data_params']['normalization'] = norm
            settings = enabled(*NAMES)
            settings['enabled'] = False
            for split, aug in [('train', None), ('train', settings),
                               ('val', enabled(*NAMES)), ('test', enabled(*NAMES))]:
                ds = self.dataset(aug, split)
                before = torch.get_rng_state().clone()
                actual = ds[0]
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assert_tensors_equal(actual, legacy_sample(ds))

    def test_rotation_common_shift_wrap_geometry_and_normalization(self):
        # Test deterministic shifts through the dataset at raw and custom paper normalization.
        for norm in ('none', {'mode': 'paper_normalization', 'mean': {'x': 3, 'y': -2},
                             'std': {'x': 2, 'y': 7}}):
            self.cfg['mos_data_params']['normalization'] = norm
            baseline = self.dataset()[0]
            ds = self.dataset(enabled('azimuth_rotation'))
            shift = 9
            raw_cfg = copy.deepcopy(self.cfg)
            raw_cfg['mos_data_params']['normalization'] = 'none'
            raw = TemporalMOSE2EDataset([self.seq], raw_cfg)[0]
            with patch('helper.mos_augmentation.torch.randint', return_value=torch.tensor(shift)):
                x, valid, dt, y, meta = ds[0]
            expected = torch.roll(raw[0], shift, -1)
            angle = -2*np.pi*shift/ds.W
            xx, yy = expected[:, 0].clone(), expected[:, 1].clone()
            expected[:, 0] = np.cos(angle)*xx - np.sin(angle)*yy
            expected[:, 1] = np.sin(angle)*xx + np.cos(angle)*yy
            spec = normalization(ds.cfg, ds.temporal_input_mode)
            expected = torch.stack([normalize_feature_tensor(f, m[None], spec)
                                   for f, m in zip(expected, torch.roll(raw[1], shift, -1))])
            torch.testing.assert_close(x, expected)
            self.assertTrue(torch.equal(valid, torch.roll(baseline[1], shift, -1)))
            self.assertTrue(torch.equal(y, torch.roll(baseline[3], shift, -1)))
            self.assertTrue(torch.equal(dt, baseline[2]))
            self.assertEqual(meta, baseline[4])
        # Explicit seam crossing and the sign of the physical XY rotation.
        f = torch.zeros(5, 1, 8); f[:, 0, 7] = torch.tensor([3., 4., 2., np.sqrt(29), .5])
        mask = torch.zeros(1, 1, 8); mask[0, 0, 7] = 1
        rotated, masks, labels = TemporalMOSAugmentation.rotate([f]*3, [mask]*3, torch.arange(8)[None], 1)
        for out in rotated:
            self.assertAlmostEqual(float(torch.atan2(out[1, 0, 0], out[0, 0, 0])), np.arctan2(4, 3)-np.pi/4, places=6)
            torch.testing.assert_close(torch.linalg.vector_norm(out[:3], dim=0), out[3])
            self.assertEqual(out[2, 0, 0], 2)
            self.assertTrue(torch.equal(out[:, 0, 1:], torch.zeros(5, 7)))
        self.assertTrue(all(m[0, 0, 0] == 1 for m in masks))
        self.assertEqual(labels[0, 0], 7)

    def test_dropout_after_alignment_before_projection_and_only_history(self):
        baseline = self.dataset()[0]
        ds = self.dataset(enabled('historical_point_dropout'))
        original = ds.augmentation.drop_history
        recorded = []
        def drop(points):
            result = original(points)
            recorded.append((points.copy(), result.copy()))
            return result
        # Force 90% of max=10%, drop/keep groups of points deterministically.
        # Randomness controls decisions, not projection or masks.
        def random(size=()):
            if size == ():
                return torch.tensor(.9)
            return (torch.arange(size).remainder(8) >= 4).float()
        with patch.object(ds.augmentation, 'applies', side_effect=lambda name: name == 'historical_point_dropout'), \
             patch.object(ds.augmentation, 'drop_history', side_effect=drop), \
             patch('helper.mos_augmentation.torch.rand', side_effect=random):
            actual = ds[0]
        self.assertEqual(len(recorded), 2)
        for t, (aligned, remaining) in enumerate(recorded):
            off = ds.offsets[t]
            expected_aligned = transform_points(ds._read_points(self.seq['paths'][2-off]),
                np.linalg.solve(self.seq['poses'][2], self.seq['poses'][2-off]))
            np.testing.assert_array_equal(aligned, expected_aligned)
            self.assertLess(len(remaining), len(aligned))
            f, mask = project_range_features(remaining, ds.H, ds.W, ds.theta_range, ds.min_range,
                                             ds.max_range, include_intensity=True, explicit_occupancy=True)
            self.assertTrue(torch.equal(actual[0][t], f))
            self.assertTrue(torch.equal(actual[1][t], mask[0].bool()))
        self.assertTrue((actual[1][:-1].sum((1, 2)) < baseline[1][:-1].sum((1, 2))).all())
        self.assertTrue(torch.equal(actual[0][-1], baseline[0][-1]))
        self.assertTrue(torch.equal(actual[1][-1], baseline[1][-1]))
        for a, b in zip(actual[2:4], baseline[2:4]):
            self.assertTrue(torch.equal(a, b))

    def test_dropout_independent_rates_and_empty_history(self):
        aug = self.dataset(enabled('historical_point_dropout')).augmentation
        points = np.arange(400, dtype=np.float32).reshape(100, 4)
        choices = torch.linspace(.005, .995, 100)
        # Event, sampled rate/max, per-point draws, independently for each scan.
        with patch('helper.mos_augmentation.torch.rand', side_effect=[
                torch.tensor(0.), torch.tensor(.2), choices,
                torch.tensor(0.), torch.tensor(.9), choices]):
            a, b = aug.drop_history(points), aug.drop_history(points)
        self.assertEqual(len(a), 98)
        self.assertEqual(len(b), 91)
        base = self.dataset()[0]
        ds = self.dataset(enabled('historical_point_dropout'))
        with patch.object(ds.augmentation, 'drop_history', side_effect=lambda points: points[:0]):
            actual = ds[0]
        self.assertFalse(actual[1][:-1].any())
        self.assertFalse(actual[0][:-1].any())
        self.assertTrue(torch.equal(actual[0][-1], base[0][-1]))
        self.assertTrue(torch.equal(actual[3], base[3]))

    def test_jitter_only_valid_intensity_before_normalization(self):
        for norm in ('none', 'intensity_only', 'paper_normalization'):
            self.cfg['mos_data_params']['normalization'] = norm
            base = self.dataset()[0]
            torch.manual_seed(11)
            actual = self.dataset(enabled('intensity_jitter'))[0]
            self.assertTrue(torch.equal(actual[0][:, :4], base[0][:, :4]))
            self.assertTrue(torch.equal(actual[0][:, 4][~base[1]], base[0][:, 4][~base[1]]))
            for t in range(3):
                self.assertTrue((actual[0][t, 4][base[1][t]] != base[0][t, 4][base[1][t]]).any())
            for a, b in zip(actual[1:4], base[1:4]):
                self.assertTrue(torch.equal(a, b))
        ds = self.dataset(enabled('intensity_jitter'))
        feature = torch.zeros(5, 1, 3); feature[4, 0] = torch.tensor([0., 1., 7.])
        with patch('helper.mos_augmentation.torch.randn', return_value=torch.tensor([-2., 2.])):
            out = ds.augmentation.jitter(feature.clone(), torch.tensor([[[1., 1., 0.]]]))
        self.assertTrue(torch.equal(out, feature))  # [0,1] clipping; invalid 7 untouched

    def test_combination_and_probability_zero(self):
        base = self.dataset()[0]
        zero = enabled(*NAMES)
        for name in NAMES:
            zero[name]['probability'] = 0
        ds = self.dataset(zero)
        before = torch.get_rng_state()
        self.assert_tensors_equal(ds[0], base)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        ds = self.dataset(enabled(*NAMES))
        for seed in range(5):
            torch.manual_seed(seed)
            x, valid, dt, y, _ = ds[0]
            self.assertEqual(x.shape, base[0].shape)
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(torch.equal(dt, base[2]))
            torch.testing.assert_close(torch.linalg.vector_norm(x[:, :3], dim=1), x[:, 3])
            self.assertTrue((y[~valid[-1]] == -1).all())
            self.assertTrue((x.permute(0, 2, 3, 1)[~valid] == 0).all())

    def test_config_validation_and_reference_hyperparameters(self):
        self.assertFalse(augmentation_settings()['enabled'])
        invalid = [[], {'enabled': 'false'}, {'typo': True}, {'azimuth_rotation': None},
                   {'azimuth_rotation': {'probability': -1}},
                   {'historical_point_dropout': {'max_dropout_ratio': 1.1}},
                   {'intensity_jitter': {'std': float('nan')}},
                   {'intensity_jitter': {'std': float('inf')}},
                   {'intensity_jitter': {'std': -1}},
                   {'intensity_jitter': {'std': True}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                augmentation_settings(value)
        with self.assertRaisesRegex(ValueError, 'requires input_mode'):
            augmentation_settings(enabled('intensity_jitter'), 'range_xyz')
        # Global disable takes precedence, including an unused intensity switch.
        augmentation_settings({**enabled('intensity_jitter'), 'enabled': False}, 'range_xyz')
        original = yaml.safe_load(REFERENCE.read_text())
        path, augmented = trainer.read_config(AUGMENTED)
        self.assertTrue(all(augmented['mos_augmentation'][n]['enabled'] for n in NAMES))
        augmented.pop('mos_augmentation')
        augmented['mos_log_params']['run_name'] = original['mos_log_params']['run_name']
        original['mos_train_params']['loss'] = trainer.loss_settings(original['mos_train_params']['loss'])
        self.assertEqual(augmented, original)

    def test_worker_reproducibility_and_distinct_samples(self):
        ds = self.dataset(enabled(*NAMES))
        for workers in (0, 2):
            def load():
                trainer.seed_all(42)
                return list(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=workers,
                            generator=torch.Generator().manual_seed(42)))
            first, again = load(), load()
            for a, b in zip(first, again):
                self.assert_tensors_equal(a, b)
            self.assertFalse(torch.equal(first[0][0], first[1][0]))

    def test_evaluator_disables_augmentation_even_for_train_split(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['mos_augmentation'] = enabled(*NAMES)
        captured = []
        def loader(cfg, *args):
            captured.append(cfg)
            return object()
        with patch.object(sys, 'argv', ['eval', '--checkpoint', str(Path(self.tmp.name)/'checkpoints/model.pt'),
                                       '--split', 'train', '--device', 'cpu']), \
             patch.object(evaluator.torch, 'load', return_value={'config': cfg, 'model_state_dict': {}, 'epoch': 1}), \
             patch.object(evaluator, 'build_mos_model', return_value=SimpleNamespace(load_state_dict=lambda *a, **k: None, to=lambda *a: None)), \
             patch.object(evaluator, 'make_loader', side_effect=loader), \
             patch.object(evaluator, 'evaluate', return_value={}):
            evaluator.main()
        self.assertEqual(captured[0]['mos_augmentation'], {'enabled': False})
        self.assertTrue(cfg['mos_augmentation']['enabled'])

    def test_benchmark_knn_threshold_inputs_ignore_augmentation(self):
        # The shared benchmark input builder supplies official evaluation,
        # point KNN, and threshold sweeps (including the raw current range).
        paths = [Path(p) for p in self.seq['paths']]
        args = (paths, 2, self.seq['poses'])
        base, boot, ranges = temporal_input(*args, self.cfg, 'strict', True,
                                           return_projected_range=True)
        cfg = copy.deepcopy(self.cfg)
        cfg['mos_augmentation'] = enabled(*NAMES)
        before = torch.get_rng_state()
        actual, actual_boot, actual_ranges = temporal_input(*args, cfg, 'strict', True,
                                                           return_projected_range=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(base, actual):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(boot, actual_boot)
        np.testing.assert_array_equal(ranges, actual_ranges)

    def test_training_saves_resolved_augmentation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['mos_augmentation'] = enabled(*NAMES)
        cfg['mos_log_params']['log_root'] = str(Path(self.tmp.name) / 'runs')
        cfg['mos_data_params']['val_sequences'] = ['08']
        val_seq = {**self.seq, 'seq_id': '08'}
        cfg_path = Path(self.tmp.name) / 'training.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg))
        args = SimpleNamespace(cfg_path=str(cfg_path), log_root=None, epochs=1,
                               batch_size=1, num_workers=0, max_batches=1, dry_run=False)
        with patch.object(trainer, 'make_sequences', return_value=[self.seq, val_seq]):
            trainer.train(args, torch.device('cpu'))
        run = next((Path(self.tmp.name) / 'runs').iterdir())
        resolved = yaml.safe_load((run / 'config.yaml').read_text())['mos_augmentation']
        self.assertEqual(resolved, augmentation_settings(cfg['mos_augmentation'], 'range_xyz_intensity'))
        saved = torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(saved['config']['mos_augmentation'], resolved)

    def test_ddp_and_workers(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['mos_augmentation'] = enabled(*NAMES)
        np.save(Path(self.tmp.name)/'seq.npy', self.seq)
        (Path(self.tmp.name)/'cfg.yaml').write_text(yaml.safe_dump(cfg))
        mp.spawn(distributed_augmentation_worker, args=(self.tmp.name,), nprocs=2, join=True)

    def test_augmented_training_both_resolutions(self):
        # Real SalsaNext + temporal attention + CE/Lovasz, batch=1, CPU FP32.
        for width in (512, 2048):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                seq, cfg = fixture(directory, 64, width)
                cfg['mos_data_params']['previous_offsets'] = [1]
                cfg['mos_data_params']['normalization'] = 'intensity_only'
                cfg['mos_augmentation'] = enabled(*NAMES)
                ds = TemporalMOSE2EDataset([seq], cfg)
                loader = torch.utils.data.DataLoader(ds, batch_size=1)
                trainer.seed_all(42)
                model = trainer.build_mos_model(cfg)
                before = model.encoder.down_cntx.conv1.weight.detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
                result = trainer.run_phase(model, loader, trainer.make_criterion(cfg, torch.device('cpu')),
                    torch.device('cpu'), optimizer, torch.amp.GradScaler('cuda', enabled=False), max_batches=1)
                self.assertTrue(np.isfinite(result['loss']))
                self.assertEqual(result['samples'], 1)
                self.assertFalse(torch.equal(before, model.encoder.down_cntx.conv1.weight))
                del optimizer, model


if __name__ == '__main__':
    unittest.main()
