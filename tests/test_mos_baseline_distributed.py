"""Residual baseline: weighted DDP gradients, exact validation and strict inputs."""
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import train_mos as trainer
from mos_dataset import MOSFrameDataset


class TinyFrames(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        x = torch.randn(6, 2, 3, generator=torch.Generator().manual_seed(index))
        y = torch.full((2, 3), int(index in (1, 2, 4)), dtype=torch.long)
        y[0, 0] = -1
        return x, y, {'frame_stem': str(index)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1)
        self.register_buffer('marker', torch.zeros(1))

    def forward(self, x):
        return self.conv(x)


def stacked(dataset, indices):
    items = [dataset[i] for i in indices]
    return torch.stack([v[0] for v in items]), torch.stack([v[1] for v in items])


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        cfg = {'mos_data_params': {'ignore_index': -1},
               'mos_train_params': {'static_class_weight': 1., 'moving_class_weight': 290.34,
                                    'loss': {'type': 'ce_lovasz'}, 'clip_grad_norm': 0}}
        dataset = TinyFrames()
        torch.manual_seed(42)
        model = trainer.DDP(TinyModel())
        reference = copy.deepcopy(trainer.unwrap_model(model))
        criterion = trainer.make_criterion(cfg, 'cpu')
        # Different class proportions across ranks, with one update.
        indices = ([0, 3], [1, 2])
        local = [stacked(dataset, ids) for ids in indices]
        x, y = torch.cat([v[0] for v in local]), torch.cat([v[1] for v in local])
        logits = reference(x)
        ce = torch.nn.functional.cross_entropy(logits, y, weight=criterion.weight, ignore_index=-1)
        lovasz = torch.stack([criterion.lovasz(part, values[1])
                              for part, values in zip(logits.chunk(2), local)]).mean()
        expected = ce + lovasz
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        expected.backward()
        reference_optimizer.step()
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices[rank]), batch_size=2)
        loss, counts, metrics = trainer.train_one_epoch(
            model, loader, criterion, torch.optim.SGD(model.parameters(), lr=.01), 'cpu', cfg)
        for actual, target in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
        assert abs(metrics['loss_ce'] - ce.item()) < 1e-6
        assert abs(metrics['loss_lovasz'] - lovasz.item()) < 1e-6
        assert abs(loss - expected.item()) < 1e-6
        assert counts['valid_pixels'] == 20
        # Validate odd-sized and single-sample datasets (empty rank included).
        for size in (5, 1):
            subset = torch.utils.data.Subset(dataset, range(size))
            loader = torch.utils.data.DataLoader(trainer.shard_validation(subset), batch_size=2)
            _, actual, _ = trainer.validate_one_epoch(model, loader, criterion, 'cpu', cfg)
            all_x, all_y = stacked(dataset, range(size))
            expected_counts = trainer.compute_mos_counts(trainer.unwrap_model(model)(all_x), all_y)
            assert actual == expected_counts, (actual, expected_counts)
        assert not any(key.startswith('module.') for key in trainer.unwrap_model(model).state_dict())
        # Exercise full DDP main: two epochs, one writer, compatible checkpoint.
        directory = Path(rendezvous).parent
        path = directory / 'ddp_config.yaml'
        if rank == 0:
            main_cfg = yaml.safe_load((ROOT / 'src/configs/mos_semanticKitti_residual1_fullres_lovasz_w10_p2_2gpu.yaml').read_text())
            main_cfg['dataset_path'] = str(directory)
            main_cfg['mos_train_params'].update(epochs=2, num_workers=0)
            main_cfg['mos_log_params'].update(log_root=str(directory), use_tensorboard=False)
            path.write_text(yaml.safe_dump(main_cfg))
        dist.barrier()
        dataset.channel_count = 6
        with patch.object(sys, 'argv', ['train_mos.py', '--cfg_path', str(path), '--device', 'cpu']), \
             patch.object(trainer, 'make_sequences', return_value=[]), \
             patch.object(trainer, 'select_sequences', return_value=[]), \
             patch.object(trainer, 'MOSFrameDataset', return_value=dataset), \
             patch.object(trainer, 'print_dataset_class_stats'), \
             patch.object(trainer, 'print_dataset_channel_debug'), \
             patch.object(trainer, 'build_mos_model', side_effect=lambda _: TinyModel()):
            trainer.main()
        if rank == 0:
            import csv
            checkpoints = list(directory.glob('*/checkpoints/last.pt'))
            assert len(checkpoints) == 1
            payload = torch.load(checkpoints[0], weights_only=False, map_location='cpu')
            assert payload['epoch'] == 2
            assert payload['cfg']['run_metadata']['global_batch_size'] == 8
            assert payload['val_counts']['valid_pixels'] == 25
            assert not any(k.startswith('module.') for k in payload['model_state_dict'])
            with (checkpoints[0].parent.parent / 'metrics.csv').open() as handle:
                assert len(list(csv.DictReader(handle))) == 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class BaselineDistributedTests(unittest.TestCase):
    def test_weighted_gradients_and_unpadded_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)

    def test_strict_residuals_and_legacy_fallback(self):
        ds = MOSFrameDataset.__new__(MOSFrameDataset)
        ds.H, ds.W, ds.strict_residuals = 2, 3, True
        ds._missing_residual_warned = set()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'residual.npy'
            with self.assertRaises(FileNotFoundError):
                ds._load_residual_image(str(p))
            np.save(p, np.zeros((1, 3), np.float32))
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                ds._load_residual_image(str(p))
            np.save(p, np.full((2, 3), np.nan, np.float32))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                ds._load_residual_image(str(p))
            np.save(p, np.ones((2, 3), np.float32))
            np.testing.assert_array_equal(ds._load_residual_image(str(p)), 1)
            ds.strict_residuals = False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.assertFalse(ds._load_residual_image(str(p.parent / 'missing.npy')).any())

    def test_config_pair_and_legacy_loss(self):
        configs = []
        for suffix in ('w10', 'w290'):
            path = ROOT / f'src/configs/mos_semanticKitti_residual1_fullres_lovasz_{suffix}_p2_2gpu.yaml'
            cfg = yaml.safe_load(path.read_text())
            self.assertEqual(cfg['mos_train_params']['batch_size'], 4)
            self.assertEqual(cfg['mos_train_params']['epochs'], 150)
            self.assertEqual(trainer.compute_in_channels(cfg['mos_data_params']['input_mode'], [1]), 6)
            self.assertEqual(cfg['mos_data_params']['residual_folder_template'], 'residual_images_{offset}_Full_Res')
            self.assertTrue(cfg['mos_data_params']['strict_residuals'])
            self.assertIsInstance(trainer.make_criterion(cfg, 'cpu'), trainer.MOSCrossEntropyLovasz)
            configs.append(cfg)
        self.assertEqual(configs[0]['mos_train_params'].pop('moving_class_weight'), 10)
        self.assertEqual(configs[1]['mos_train_params'].pop('moving_class_weight'), 290.34)
        for cfg in configs:
            cfg['mos_log_params'].pop('run_name')
        self.assertEqual(*configs)
        legacy = {'mos_train_params': {'static_class_weight': 1, 'moving_class_weight': 10}}
        self.assertIs(type(trainer.make_criterion(legacy, 'cpu')), torch.nn.CrossEntropyLoss)

    def test_submit_gpu_gates_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / 'sbatch'
            fake.write_text("#!/usr/bin/env python3\n"
                            "import json, os, pathlib, sys\n"
                            "p=pathlib.Path(os.environ['SUBMIT_RECORD'])\n"
                            "n=len(p.read_text().splitlines()) if p.exists() else 0\n"
                            "with p.open('a') as f: f.write(json.dumps({'args':sys.argv[1:], 'dependency':os.environ.get('SBATCH_DEPENDENCY')})+'\\n')\n"
                            "print(str(100+n)+';cluster')\n")
            fake.chmod(0o755)
            record = directory / 'jobs.jsonl'
            env = {**os.environ, 'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                   'HPC_ROOT': str(directory), 'SUBMIT_RECORD': str(record)}
            env.pop('SBATCH_DEPENDENCY', None)
            env.pop('P2_GPU_GRES', None)
            subprocess.run(['bash', str(ROOT / 'submit_mos_residual_comparison_p2.sh')],
                           env=env, check=True, capture_output=True, text=True)
            jobs = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertIn('--gres=gpu:a100_40gb:2', job['args'])
                self.assertIn('--partition=p2', job['args'])
                self.assertTrue(any('train_mos_residual_p2_2gpu.slurm' in arg for arg in job['args']))
            for job in jobs[:2]:
                self.assertIn('--smoke_test', job['args'])
            for job in jobs[2:]:
                self.assertNotIn('--smoke_test', job['args'])
                self.assertEqual(job['dependency'], 'afterok:100:101')

    def test_main_checkpoint_and_loss_logging(self):
        import csv
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            config_path = ROOT / 'src/configs/mos_semanticKitti_residual1_fullres_lovasz_w10_p2_2gpu.yaml'
            cfg = yaml.safe_load(config_path.read_text())
            cfg['dataset_path'] = directory
            cfg['mos_train_params'].update(epochs=1, num_workers=0)
            cfg['mos_log_params'].update(log_root=directory, use_tensorboard=False)
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            data = TinyFrames()
            data.channel_count = 6
            # Main orchestration, real losses/optimizer/CSV/checkpoints, synthetic data.
            with patch.object(sys, 'argv', ['train_mos.py', '--cfg_path', str(path), '--device', 'cpu']), \
                 patch.object(trainer, 'make_sequences', return_value=[]), \
                 patch.object(trainer, 'select_sequences', return_value=[]), \
                 patch.object(trainer, 'MOSFrameDataset', return_value=data), \
                 patch.object(trainer, 'print_dataset_class_stats'), \
                 patch.object(trainer, 'print_dataset_channel_debug'), \
                 patch.object(trainer, 'build_mos_model', side_effect=lambda _: TinyModel()):
                trainer.main()
            checkpoint_path = next(Path(directory).glob('*/checkpoints/best_moving_iou.pt'))
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['cfg']['run_metadata']['global_batch_size'], 4)
            self.assertFalse(any(k.startswith('module.') for k in checkpoint['model_state_dict']))
            with (checkpoint_path.parent.parent / 'metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            for phase in ('train', 'val'):
                self.assertAlmostEqual(float(rows[0][phase + '_loss_total']),
                                       float(rows[0][phase + '_loss_ce']) + float(rows[0][phase + '_loss_lovasz']))


if __name__ == '__main__':
    unittest.main()
