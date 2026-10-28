"""P1 residual configs, AMP orchestration and FP16-safe CE normalization."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from test_mos_baseline_distributed import ROOT, TinyFrames, TinyModel, trainer


class HalfOutput(TinyModel):
    def forward(self, x):
        return super().forward(x).half()


class BaselineAMPTests(unittest.TestCase):
    def test_p1_configs_preserve_experiment(self):
        for suffix in ('w10', 'w290'):
            p2 = yaml.safe_load((ROOT / f'src/configs/mos_semanticKitti_residual1_fullres_lovasz_{suffix}_p2_2gpu.yaml').read_text())
            p1 = yaml.safe_load((ROOT / f'src/configs/mos_semanticKitti_residual1_fullres_lovasz_{suffix}_p1_2gpu_amp.yaml').read_text())
            self.assertTrue(p1['mos_train_params']['amp'])
            self.assertEqual(p1['mos_train_params']['amp_dtype'], 'float16')
            self.assertEqual(p1['mos_train_params']['batch_size'], 4)
            p1['mos_train_params']['amp'] = False
            p1['mos_train_params'].pop('amp_dtype')
            p1['mos_log_params']['run_name'] = p2['mos_log_params']['run_name']
            self.assertEqual(p1, p2)

    def test_amp_device_and_dtype_validation(self):
        cfg = {'mos_train_params': {'amp': True, 'amp_dtype': 'float16'}}
        self.assertEqual(trainer.amp_settings(cfg, 'cuda:0'), (True, torch.float16))
        self.assertEqual(trainer.amp_settings(cfg, 'cpu'), (False, torch.float16))
        cfg['mos_train_params']['amp_dtype'] = 'bfloat16'
        with patch.object(torch.cuda, 'is_bf16_supported', return_value=False):
            with self.assertRaisesRegex(ValueError, 'does not support bfloat16'):
                trainer.amp_settings(cfg, 'cuda:0')

    def test_fp16_outputs_do_not_overflow_ce_denominator(self):
        torch.set_num_threads(1)
        cfg = {'mos_train_params': {'static_class_weight': 1., 'moving_class_weight': 290.34,
                                    'loss': {'type': 'ce_lovasz'}}}
        criterion = trainer.make_criterion(cfg, 'cpu')
        original = criterion.components
        # CPU simulation of CUDA autocast's FP32 CE path with FP16 logits.
        criterion.components = lambda logits, y: original(logits.float(), y)
        model = HalfOutput()
        before = model.conv.weight.detach().clone()
        x = torch.randn(2, 6, 16, 32)
        y = torch.ones(2, 16, 32, dtype=torch.long)
        self.assertGreater(criterion.weight[y].sum().item(), torch.finfo(torch.float16).max)
        loader = [(x, y, {})]
        loss, _, metrics = trainer.train_one_epoch(model, loader, criterion,
                            torch.optim.SGD(model.parameters(), lr=.01), 'cpu', cfg)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(metrics['optimizer_steps'], 1)
        self.assertFalse(torch.equal(before, model.conv.weight))

    def test_cpu_autocast_keeps_lovasz_fp32(self):
        torch.set_num_threads(1)
        cfg = {'mos_train_params': {'static_class_weight': 1., 'moving_class_weight': 10.,
                                    'loss': {'type': 'ce_lovasz'}, 'clip_grad_norm': 5}}
        criterion = trainer.make_criterion(cfg, 'cpu')
        model = TinyModel()
        dtypes = []
        handle = criterion.lovasz.register_forward_hook(lambda _module, _args, output: dtypes.append(output.dtype))
        with patch.object(trainer, 'amp_settings', return_value=(True, torch.bfloat16)):
            loss, _, metrics = trainer.train_one_epoch(model,
                torch.utils.data.DataLoader(TinyFrames(), batch_size=2), criterion,
                torch.optim.SGD(model.parameters(), lr=.01), 'cpu', cfg)
        handle.remove()
        self.assertEqual(set(dtypes), {torch.float32})
        self.assertEqual(metrics['optimizer_steps'], 3)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_p1_submit_only_smoke_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / 'sbatch'
            fake.write_text("#!/usr/bin/env python3\n"
                            "import json, os, pathlib, sys\n"
                            "with pathlib.Path(os.environ['RECORD']).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                            "print('123')\n")
            fake.chmod(0o755)
            record = directory / 'record'
            env = {**os.environ, 'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                   'HPC_ROOT': str(directory), 'RECORD': str(record)}
            for suffix in ('w10', 'w290'):
                subprocess.run(['bash', str(ROOT / 'submit_mos_residual_p1_2gpu.sh'),
                    f'src/configs/mos_semanticKitti_residual1_fullres_lovasz_{suffix}_p1_2gpu_amp.yaml',
                    '--smoke_test'], env=env, check=True, capture_output=True)
            jobs = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                self.assertIn('--partition=p1', job)
                self.assertIn('--gres=gpu:2', job)
                self.assertIn('--smoke_test', job)
                self.assertTrue(any('train_mos_residual_p1_2gpu.slurm' in arg for arg in job))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fp16_optimizer_update(self):
        cfg = {'mos_train_params': {'static_class_weight': 1., 'moving_class_weight': 290.34,
               'loss': {'type': 'ce_lovasz'}, 'amp': True, 'amp_dtype': 'float16', 'clip_grad_norm': 5},
               'run_metadata': {'smoke_test': True}}
        model = TinyModel().cuda()
        before = copy.deepcopy(model.state_dict())
        criterion = trainer.make_criterion(cfg, 'cuda')
        x = torch.randn(4, 6, 16, 32)
        y = torch.ones(4, 16, 32, dtype=torch.long)
        scaler = trainer.make_grad_scaler(cfg, 'cuda')
        loss, _, metrics = trainer.train_one_epoch(model, [(x,y,{})], criterion,
            torch.optim.AdamW(model.parameters(), lr=.0005), 'cuda', cfg, scaler=scaler)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(metrics['optimizer_steps'], 1)
        self.assertTrue(scaler.is_enabled())
        self.assertFalse(torch.equal(before['conv.weight'], model.conv.weight))


if __name__ == '__main__':
    unittest.main()
