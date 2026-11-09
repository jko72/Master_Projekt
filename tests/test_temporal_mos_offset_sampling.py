"""Variable-offset geometry, RNG, workers, DDP, logging and checkpoint checks."""
from copy import deepcopy
import csv
import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from helper.temporal_offset_sampling import validate_temporal_offset_sampling
from helper.residual_helper import transform_points
from helper.mos_projection import project_features
from helper.mos_input import normalization, normalize_feature_tensor
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
import train_temporal_mos_e2e as trainer
from evaluate_mos_benchmark import temporal_input
import evaluate_temporal_mos_e2e as evaluator

BASE = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate.yaml'
EXPERIMENT = BASE.with_name(BASE.stem + '_temporal_sampling.yaml')
CPU = torch.device('cpu')


def fixture(root, seq_id='00', count=24):
    directory = Path(root)/seq_id
    (directory/'velodyne').mkdir(parents=True)
    if int(seq_id) < 11:
        (directory/'labels').mkdir()
    paths, poses = [], []
    for i in range(count):
        # Different raw geometry AND remission per frame, with moving/static GT.
        points = np.array([[10+i*.02, 0, -.5, .01*i], [0, 8, -.4, .5],
                           [-6, 2, -.3, .2], [0, -5, -.2, .3]], np.float32)
        path = directory/'velodyne'/f'{i:06d}.bin'
        points.tofile(path)
        if int(seq_id) < 11:
            np.array([10, 252, 40, 251], np.uint32).tofile(directory/'labels'/f'{i:06d}.label')
        angle = .01*i
        pose = np.eye(4)
        pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        pose[0, 3] = .1*i
        paths.append(str(path))
        poses.append(pose)
    return dict(seq_id=seq_id, paths=paths, poses=poses)


def config():
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    cfg['model_params'].update(grid_height=16, grid_width=64)
    cfg['mos_augmentation'] = {'enabled': False}
    cfg['mos_log_params']['use_tensorboard'] = False
    return cfg


def offset_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous', rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=120))
    try:
        seq = np.load(Path(directory)/'seq.npy', allow_pickle=True).item()
        cfg = config()
        cfg['mos_data_params']['train_sequences'] = ['00']
        def load():
            trainer.seed_all(42 + rank)
            with patch.object(trainer, 'make_sequences', return_value=[seq]):
                loader = trainer.make_loader(cfg, 'train', 4, 2, CPU)
                assert isinstance(loader.sampler, torch.utils.data.DistributedSampler)
                loader.sampler.set_epoch(3)
                return loader
        first = [b[-1]['sampled_previous_offset'].tolist() for b in load()]
        again = [b[-1]['sampled_previous_offset'].tolist() for b in load()]
        assert first == again
        loader = load()
        model = trainer.wrap_distributed(trainer.build_mos_model(cfg), CPU)
        before = model.module.encoder.down_cntx.conv1.weight.detach().clone()
        optimizer = trainer.make_optimizer(model.parameters(), cfg['mos_train_params'])
        log = io.StringIO()
        result = trainer.run_phase(model, loader, trainer.make_criterion(cfg, CPU), CPU,
            optimizer, torch.amp.GradScaler('cuda', enabled=False), max_batches=2, sample_log=log)
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        gathered = [None, None]
        dist.all_gather_object(gathered, records)
        all_records = sum(gathered, [])
        assert result['samples'] == result['offset_sample_count'] == len(all_records) == 16
        assert result['missing_history_count'] == sum(not r['history_available'][0] for r in all_records)
        for off in (5, 8, 10):
            assert result[f'offset_{off}_count'] == sum(r['sampled_previous_offset'] == off for r in all_records)
        assert abs(sum(result[f'offset_{off}_ratio'] for off in (5, 8, 10)) - 1) < 1e-12
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert not torch.equal(before, model.module.encoder.down_cntx.conv1.weight)
    finally:
        dist.destroy_process_group()


class OffsetSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mos-offset-')
        self.addCleanup(self.tmp.cleanup)
        self.seq = fixture(self.tmp.name)
        self.cfg = config()

    def dataset(self, cfg=None, split='train', sequences=None):
        return TemporalMOSE2EDataset(sequences or [self.seq], cfg or self.cfg, split)

    def test_config_is_exact_copy_except_sampling_and_name(self):
        original = yaml.safe_load(BASE.read_text())
        new = yaml.safe_load(EXPERIMENT.read_text())
        sampling = new['mos_data_params'].pop('temporal_offset_sampling')
        self.assertEqual(sampling, dict(enabled=True, offsets=[5,8,10], probabilities=[.25,.5,.25], train_only=True))
        self.assertEqual(new['mos_data_params']['previous_offsets'], [8])
        self.assertEqual(new['mos_log_params']['run_name'], 'MOS_temporal_e2e_lmnet_cb_gate_temporal_sampling_5_8_10_w1')
        new['mos_log_params']['run_name'] = original['mos_log_params']['run_name']
        self.assertEqual(new, original)
        trainer.read_config(EXPERIMENT)

    def test_disabled_exact_inputs_and_rng_with_existing_augmentation(self):
        legacy = deepcopy(self.cfg)
        legacy['mos_data_params'].pop('temporal_offset_sampling')
        disabled = deepcopy(self.cfg)
        disabled['mos_data_params']['temporal_offset_sampling']['enabled'] = False
        for aug in ({'enabled': False}, {'mode': 'lmnet'}):
            legacy['mos_augmentation'] = disabled['mos_augmentation'] = aug
            old, new = self.dataset(legacy), self.dataset(disabled)
            for index in (3, 9, 15):
                trainer.seed_all(42)
                expected = old[index]
                rng = torch.get_rng_state().clone()
                trainer.seed_all(42)
                actual = new[index]
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for a, b in zip(actual[:-1], expected[:-1]):
                    self.assertTrue(torch.equal(a, b))
                self.assertEqual(actual[-1], expected[-1])
                self.assertEqual(actual[-1]['previous_offsets'], [8])
        self.assertNotIn('temporal_offset_sampling', legacy['mos_data_params'])

    def test_forced_offsets_real_scans_poses_projection_and_current_targets(self):
        fixed = deepcopy(self.cfg)
        fixed['mos_data_params'].pop('temporal_offset_sampling')
        baseline = self.dataset(fixed)[15]
        for off in (5, 8, 10):
            cfg = deepcopy(self.cfg)
            cfg['mos_data_params']['temporal_offset_sampling']['probabilities'] = [float(o == off) for o in (5,8,10)]
            ds = self.dataset(cfg)
            with patch('helper.temporal_e2e_input.transform_points', wraps=transform_points) as transform:
                actual = ds[15]
            x, valid, dt, mos, movable, meta = actual
            expected_path = self.seq['paths'][15-off]
            points = ds._read_points(expected_path)
            pose_transform = np.linalg.solve(self.seq['poses'][15], self.seq['poses'][15-off])
            np.testing.assert_array_equal(transform.call_args.args[0], points)
            np.testing.assert_allclose(transform.call_args.args[1], pose_transform)
            features, mask, _ = project_features(transform_points(points, pose_transform), 16, 64, ds.theta_range)
            expected = normalize_feature_tensor(features, mask, normalization(cfg, ds.temporal_input_mode))
            torch.testing.assert_close(x[0], expected)
            torch.testing.assert_close(valid[0], mask[0].bool())
            torch.testing.assert_close(dt, torch.tensor([-off/10, 0.]))
            self.assertEqual(x.shape, (2,5,16,64))
            self.assertEqual(valid.shape, (2,16,64))
            self.assertEqual(meta['sampled_previous_offset'], off)
            self.assertEqual(meta['history_paths'], [expected_path])
            self.assertEqual(meta['history_available'], [True])
            self.assertEqual(ds.offsets, [8])
            for a, b in ((x[-1], baseline[0][-1]), (mos, baseline[3]), (movable, baseline[4])):
                self.assertTrue(torch.equal(a, b))
            if off == 8:
                for a, b in zip(actual[:-1], baseline[:-1]):
                    self.assertTrue(torch.equal(a, b))

    def test_early_frames_missing_scans_poses_and_sequence_boundaries(self):
        ds = self.dataset()
        self.assertEqual(len(ds), len(self.seq['paths']))
        early = ds[3]
        self.assertEqual(early[-1]['sampled_previous_offset'], 8)
        self.assertEqual(early[-1]['history_available'], [False])
        self.assertFalse(early[0][0].any())
        self.assertFalse(early[1][0].any())
        torch.testing.assert_close(early[2], torch.tensor([-.8,0.]))
        self.assertEqual(ds[6][-1]['sampled_previous_offset'], 5)
        seen = {ds._sample_previous_offset(ds.samples[9]) for _ in range(300)}
        self.assertEqual(seen, {5,8})
        seen = {ds._sample_previous_offset(ds.samples[10]) for _ in range(300)}
        self.assertEqual(seen, {5,8,10})
        # A missing file and an invalid historical pose cannot be selected.
        Path(self.seq['paths'][7]).unlink()  # t12-5
        ds.sequence_map['00']['poses'][4] = np.full((4,4), np.nan)  # t12-8
        self.assertEqual(ds[12][-1]['sampled_previous_offset'], 10)
        Path(self.seq['paths'][2]).unlink()
        sample = ds[12]
        self.assertEqual(sample[-1]['sampled_previous_offset'], 8)
        self.assertEqual(sample[-1]['history_available'], [False])
        self.assertFalse(sample[0][0].any())
        self.assertFalse(sample[1][0].any())
        other = fixture(self.tmp.name, '01', count=4)
        combined = self.dataset(sequences=[self.seq, other])
        self.assertEqual(len(combined), 28)
        self.assertEqual(combined[-1][-1]['history_available'], [False])
        # A shorter pose array is unavailable too, without crossing sequences.
        seq = deepcopy(other)
        seq['poses'] = seq['poses'][:2]
        self.assertIsNone(ds._lmnet_history_path(seq, 2))
        # If the reference is outside the candidate list it must still stay
        # empty when no candidate exists, even if the reference scan exists.
        cfg = deepcopy(self.cfg)
        cfg['mos_data_params']['temporal_offset_sampling'].update(offsets=[10], probabilities=[1.])
        sample = self.dataset(cfg)[9]
        self.assertEqual(sample[-1]['sampled_previous_offset'], 8)
        self.assertEqual(sample[-1]['history_available'], [False])
        self.assertFalse(sample[1][0].any())

    def test_probabilities_and_conditional_renormalization(self):
        ds = self.dataset()
        trainer.seed_all(42)
        full = [ds._sample_previous_offset(ds.samples[15]) for _ in range(12000)]
        for off, expected in zip((5,8,10), (.25,.5,.25)):
            self.assertAlmostEqual(full.count(off)/len(full), expected, delta=.02)
        partial = [ds._sample_previous_offset(ds.samples[9]) for _ in range(6000)]
        self.assertAlmostEqual(partial.count(5)/len(partial), 1/3, delta=.025)
        # A valid config can leave only zero-weight histories at sequence start.
        cfg = deepcopy(self.cfg)
        cfg['mos_data_params']['temporal_offset_sampling']['probabilities'] = [0.,0.,1.]
        self.assertEqual(self.dataset(cfg)[6][-1]['sampled_previous_offset'], 5)

    def test_validation_test_and_benchmark_always_reference_without_rng(self):
        for split in ('val', 'test'):
            seq = self.seq if split == 'val' else fixture(self.tmp.name, '11')
            ds = self.dataset(split=split, sequences=[seq])
            before = torch.get_rng_state().clone()
            sample = ds[15]
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(sample[-1]['previous_offsets'], [8])
            torch.testing.assert_close(sample[2], torch.tensor([-.8,0.]))
        paths = [Path(p) for p in self.seq['paths']]
        fixed = deepcopy(self.cfg)
        fixed['mos_data_params'].pop('temporal_offset_sampling')
        before = torch.get_rng_state().clone()
        variable, boot = temporal_input(paths, 15, self.seq['poses'], self.cfg, 'strict', True)
        reference, fixed_boot = temporal_input(paths, 15, self.seq['poses'], fixed, 'strict', True)
        self.assertEqual(boot, fixed_boot)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(variable, reference):
            self.assertTrue(torch.equal(a, b))
        torch.testing.assert_close(variable[2], torch.tensor([[-.8,0.]]))

    def test_workers_seed_reproducibility_and_immutable_offsets(self):
        ds = self.dataset()
        for workers in (0,2):
            def load(seed):
                trainer.seed_all(seed)
                loader = torch.utils.data.DataLoader(ds, batch_size=3, num_workers=workers,
                    generator=torch.Generator().manual_seed(seed))
                return torch.cat([b[-1]['sampled_previous_offset'] for b in loader]).tolist()
            self.assertEqual(load(42), load(42))
            self.assertNotEqual(load(42), load(2026))
            self.assertEqual(ds.offsets, [8])

    def test_actual_model_movable_forward_backward_with_variable_dt(self):
        model = trainer.build_mos_model(self.cfg)
        criterion = trainer.make_criterion(self.cfg, CPU)
        ds = self.dataset()
        samples = []
        for off in (5,8,10):
            with patch.object(ds, '_sample_previous_offset', return_value=off):
                samples.append(ds[15])
        x, valid, dt, mos, movable, _ = trainer.collate_mos(samples)
        output, aux = model(x, valid, dt, return_aux=True)
        self.assertEqual(output.shape, (3,2,16,64))
        self.assertEqual(aux.shape, output.shape)
        loss = criterion(output, mos) + criterion.movable_criterion(aux, movable)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        fixed = deepcopy(self.cfg)
        fixed['mos_data_params'].pop('temporal_offset_sampling')
        self.assertEqual({k: v.shape for k,v in model.state_dict().items()},
                         {k: v.shape for k,v in trainer.build_mos_model(fixed).state_dict().items()})

    def test_end_to_end_logging_checkpoint_and_resume(self):
        cfg = deepcopy(self.cfg)
        cfg['mos_log_params']['log_root'] = str(Path(self.tmp.name)/'runs')
        cfg['mos_train_params']['write_sample_logs'] = True
        cfg['mos_data_params'].update(train_sequences=['00'], val_sequences=['08'])
        val = fixture(self.tmp.name, '08')
        path = Path(self.tmp.name)/'config.yaml'
        path.write_text(yaml.safe_dump(cfg))
        args = SimpleNamespace(cfg_path=str(path), log_root=None, epochs=1, batch_size=2,
                               num_workers=0, max_batches=2, dry_run=False)
        with patch.object(trainer, 'make_sequences', return_value=[self.seq,val]):
            trainer.train(args, CPU)
        run = next((Path(self.tmp.name)/'runs').iterdir())
        saved = torch.load(run/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        resolved = yaml.safe_load((run/'config.yaml').read_text())
        self.assertEqual(saved['config']['mos_data_params'], resolved['mos_data_params'])
        self.assertEqual(saved['config']['mos_data_params']['temporal_offset_sampling'],
                         cfg['mos_data_params']['temporal_offset_sampling'])
        self.assertIn('torch', saved['rng_states'][0])
        self.assertIn('loader', saved['rng_states'][0])
        row = next(csv.DictReader((run/'csv/epochs.csv').read_text().splitlines()))
        self.assertEqual(int(row['train_offset_sample_count']), 4)
        self.assertEqual(sum(int(row[f'train_offset_{o}_count']) for o in (5,8,10)), 4)
        self.assertFalse(any(k.startswith('val_offset_') for k in row))
        records = [json.loads(line) for line in (run/'samples_rank0.jsonl').read_text().splitlines()]
        train = [r for r in records if r['phase'] == 'train']
        self.assertEqual(int(row['train_missing_history_count']), sum(not r['history_available'][0] for r in train))
        self.assertTrue(all(r['sampled_previous_offset'] == 8 for r in records if r['phase'] == 'val'))
        # Matching sampling config resumes a second epoch using the saved states.
        args.resume = str(run/'checkpoints/last.pt')
        args.epochs = 2
        args.log_root = str(Path(self.tmp.name)/'resume')
        with patch.object(trainer, 'make_sequences', return_value=[self.seq,val]):
            trainer.train(args, CPU)
        resumed = next((Path(self.tmp.name)/'resume').iterdir())
        resumed_row = next(csv.DictReader((resumed/'csv/epochs.csv').read_text().splitlines()))
        self.assertEqual(resumed_row['epoch'], '2')
        self.assertEqual(resumed_row['train_offset_sample_count'], '4')
        # Fixed-offset legacy checkpoint must not resume into variable sampling.
        saved['config']['mos_data_params'].pop('temporal_offset_sampling')
        legacy = Path(self.tmp.name)/'fixed.pt'
        torch.save(saved, legacy)
        args.resume = str(legacy)
        args.log_root = str(Path(self.tmp.name)/'mismatch')
        with patch.object(trainer, 'make_sequences', return_value=[self.seq,val]), \
             self.assertRaisesRegex(ValueError, 'mos_data_params'):
            trainer.train(args, CPU)
        # Old configs/checkpoints remain usable together, with no sampling columns.
        cfg['mos_data_params'].pop('temporal_offset_sampling')
        path.write_text(yaml.safe_dump(cfg))
        args.log_root = str(Path(self.tmp.name)/'legacy-resume')
        with patch.object(trainer, 'make_sequences', return_value=[self.seq,val]):
            trainer.train(args, CPU)
        legacy_run = next((Path(self.tmp.name)/'legacy-resume').iterdir())
        legacy_row = next(csv.DictReader((legacy_run/'csv/epochs.csv').read_text().splitlines()))
        self.assertFalse(any(k.startswith('train_offset_') for k in legacy_row))

    def test_evaluation_on_train_split_disables_sampling(self):
        captured = []
        def loader(cfg, *args):
            captured.append(cfg)
            return object()
        with patch.object(sys, 'argv', ['eval', '--checkpoint', str(Path(self.tmp.name)/'checkpoints/model.pt'),
                                      '--split', 'train', '--device', 'cpu']), \
             patch.object(evaluator.torch, 'load', return_value={'config': self.cfg, 'model_state_dict': {}, 'epoch': 1}), \
             patch.object(evaluator, 'build_mos_model', return_value=SimpleNamespace(
                 load_state_dict=lambda *a, **k: None, to=lambda *a: None)), \
             patch.object(evaluator, 'make_loader', side_effect=loader), \
             patch.object(evaluator, 'evaluate', return_value={}):
            evaluator.main()
        self.assertFalse(captured[0]['mos_data_params']['temporal_offset_sampling']['enabled'])
        self.assertTrue(self.cfg['mos_data_params']['temporal_offset_sampling']['enabled'])

    def test_invalid_settings_and_legacy_compatibility(self):
        settings = self.cfg['mos_data_params']['temporal_offset_sampling']
        invalid = [[], None, {'enabled': 'true'}, {'enabled': 1}, {'train_only': False},
                   {'train_only': 1}, {'offsets': []}, {'offsets': [0,8]}, {'offsets': [-1,8]},
                   {'offsets': [5,5]}, {'offsets': [True,8]}, {'offsets': [5.0,8]},
                   {'offsets': '5,8'}, {'probabilities': [.25,.5]},
                   {'probabilities': [.25,.5,.5]}, {'probabilities': [-.25,1.,.25]},
                   {'probabilities': [0.,0.,0.]}, {'probabilities': [float('nan'),.5,.5]},
                   {'probabilities': [float('inf'),.5,.5]}, {'probabilities': 'bad'},
                   {'probabilities': [True,0.,0.]}, {'probabilities': ['.25',.5,.25]},
                   {'unknown': True}]
        for changes in invalid:
            cfg = deepcopy(self.cfg)
            cfg['mos_data_params']['temporal_offset_sampling'] = ({**settings, **changes} if isinstance(changes,dict) else changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_temporal_offset_sampling(cfg)
        for reference in ([], [5,8], [True], [0]):
            cfg = deepcopy(self.cfg)
            cfg['mos_data_params']['previous_offsets'] = reference
            with self.assertRaisesRegex(ValueError, 'previous_offsets'):
                validate_temporal_offset_sampling(cfg)
        cfg = deepcopy(self.cfg)
        cfg['mos_auxiliary']['task'] = 'residual'
        with self.assertRaisesRegex(ValueError, 'residual'):
            validate_temporal_offset_sampling(cfg)
        cfg = deepcopy(self.cfg)
        cfg['mos_data_params']['projection_mode'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'LMNet'):
            validate_temporal_offset_sampling(cfg)
        cfg['mos_data_params'].pop('temporal_offset_sampling')
        original = deepcopy(cfg)
        self.assertFalse(validate_temporal_offset_sampling(cfg)['enabled'])
        self.assertEqual(cfg, original)
        # read_config invokes the same strict validator.
        cfg = deepcopy(self.cfg)
        cfg['mos_data_params']['temporal_offset_sampling']['typo'] = 1
        path = Path(self.tmp.name)/'invalid.yaml'
        path.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            trainer.read_config(path)

    def test_two_rank_gloo_training_workers_and_global_statistics(self):
        np.save(Path(self.tmp.name)/'seq.npy', self.seq)
        mp.spawn(offset_ddp_worker, args=(self.tmp.name,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
