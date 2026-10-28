"""CPU invariants for E2E masked reconstruction, full transfer and resume."""
from copy import deepcopy
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
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tools'))
from test_temporal_mos_augmentation import fixture
import train_temporal_e2e_mae as mae
import train_temporal_mos_e2e as mos
from helper.temporal_e2e_transfer import HEAD, load_mae_weights, validate_mae_checkpoint
from temporal_e2e_mae_dataset import TemporalE2EMAEDataset, masked_input, mask_settings, patch_mask
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from loss_temporal_e2e_mae import reconstruction_loss, reconstruction_settings

CONFIG = ROOT/'src/configs/semanticKitti_temporal_e2e_mae_intensity_off8_64x512_aug.yaml'
BASELINE = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_64x512_lovasz_w290_aug.yaml'
FINETUNE = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_mae_intensity_off8_64x512_lovasz_w290_aug.yaml'


def local_config(directory):
    seq, reference = fixture(directory, height=16, width=64)
    _, cfg = mae.read_config(CONFIG)
    cfg['model_params'] = reference['model_params']
    cfg['mos_data_params'] = reference['mos_data_params']
    cfg['mos_data_params'].update(val_sequences=['08'])
    cfg['mos_augmentation']['enabled'] = False
    cfg['mae_train_params'].update(epochs=3, batch_size=1, num_workers=0, amp=False)
    cfg['mae_log_params'].update(log_root=str(Path(directory)/'runs'), use_tensorboard=False)
    return seq, cfg, reference


def ddp_check(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+str(Path(directory)/'init'), rank=rank, world_size=2)
    try:
        seq = np.load(Path(directory)/'seq.npy', allow_pickle=True).item()
        cfg = yaml.safe_load((Path(directory)/'cfg.yaml').read_text())
        cfg['mae_train_params']['num_workers'] = 1
        with patch.object(mos, 'make_sequences', return_value=[seq, {**seq, 'seq_id': '08'}]):
            loaders = [mae.make_loader(cfg, split, torch.device('cpu')) for split in ('train', 'val')]
        mos.seed_all(42+rank)
        model = mos.wrap_distributed(mos.build_mos_model(cfg), torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        for epoch in (0, 1):
            for loader in loaders:
                mae.set_epoch(loader, epoch)
            result = mae.phase(model, loaders[0], cfg, torch.device('cpu'), optimizer, scaler, max_batches=1)
            assert np.isfinite(result['loss_total'])
            assert not any(mos.gradient_health(model).values())
            val = mae.phase(model, loaders[1], cfg, torch.device('cpu'), max_batches=1)
            assert np.isfinite(val['loss_total'])
            values = [None, None]
            dist.all_gather_object(values, val)
            assert values[0] == values[1]
        # Per-rank RNG capture must support the payload used for resume.
        states = [None, None]
        dist.all_gather_object(states, mae.capture_rng(loaders))
        payload = mae.checkpoint_payload(model, optimizer, mae.make_scheduler(optimizer, cfg['mae_train_params']),
                                         scaler, cfg, 2, val['loss_total'], states)
        assert len(payload['rng_by_rank']) == 2
    finally:
        dist.destroy_process_group()


class TemporalE2EMAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='e2e-mae-test-')
        self.addCleanup(self.tmp.cleanup)
        self.seq, self.cfg, self.mos_cfg = local_config(self.tmp.name)
        mos.seed_all(42)

    def batch(self, cfg=None):
        ds = TemporalE2EMAEDataset([self.seq], cfg or self.cfg)
        return next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))

    def model(self):
        return mos.build_mos_model(self.cfg)

    def test_dynamic_frames_heads_and_state_keys(self):
        model = self.model()
        for frames in (2, 3, 4):
            x = torch.randn(1, frames, 5, 16, 32)
            valid = torch.ones(1, frames, 16, 32, dtype=torch.bool)
            dt = torch.tensor([[-.1*offset for offset in range(frames-1, -1, -1)]])
            self.assertEqual(model(x, valid, dt).shape, (1, 5, 16, 32))
        mos_cfg = deepcopy(self.cfg)
        mos_cfg['mos_model_params']['num_classes'] = 2
        state = mos.build_mos_model(mos_cfg).state_dict()
        self.assertEqual(set(state), set(model.state_dict()))
        self.assertEqual({k for k, v in model.state_dict().items() if v.shape != state[k].shape}, HEAD)
        for offsets in ([5], [5, 1], [5, 3, 1]):
            cfg = deepcopy(self.cfg)
            cfg['mos_data_params']['previous_offsets'] = offsets
            sample = TemporalE2EMAEDataset([self.seq], cfg)[0]
            self.assertEqual(sample['x'].shape[0], len(offsets)+1)
            torch.testing.assert_close(sample['dt'], torch.tensor([-o/10 for o in offsets]+[0.]))

    def test_label_independence_and_same_mos_inputs(self):
        baseline = TemporalMOSE2EDataset([self.seq], self.mos_cfg)[0]
        # The label-free loader must never call label loaders or read label stats.
        with patch.object(TemporalMOSE2EDataset, '_load_mos_label', side_effect=AssertionError('label read')):
            sample = TemporalE2EMAEDataset([self.seq], self.cfg)[0]
        for key, expected in zip(('x', 'valid', 'dt'), baseline[:3]):
            self.assertTrue(torch.equal(sample[key], expected))
        self.assertTrue(torch.equal(sample['target_current'], sample['x'][-1]))
        for path in (Path(self.tmp.name)/'00/mos_labels').glob('*.npy'):
            path.unlink()
        self.assertGreater(len(TemporalE2EMAEDataset([self.seq], self.cfg)), 0)
        with self.assertRaisesRegex(ValueError, 'test'):
            TemporalE2EMAEDataset([self.seq], self.cfg, split='test')
        with self.assertRaisesRegex(ValueError, 'Test sequence'):
            TemporalE2EMAEDataset([{**self.seq, 'seq_id':'11'}], self.cfg)
        # Same augmentation draws must yield the same augmented inputs in both tasks.
        aug_cfg = deepcopy(self.cfg)
        aug_cfg['mos_augmentation']['enabled'] = True
        aug_ds = TemporalE2EMAEDataset([self.seq], aug_cfg)
        self.assertTrue(torch.isfinite(aug_ds[0]['x']).all())

    def test_masking_original_validity_no_target_or_skip_leak(self):
        batch = self.batch()
        x, mask, valid = batch['x'], batch['mask'], batch['valid']
        masked = masked_input(x, mask)
        self.assertTrue(torch.equal(masked[:, :-1], x[:, :-1]))
        self.assertTrue((masked[:, -1].permute(0, 2, 3, 1)[mask] == 0).all())
        self.assertTrue(torch.equal(masked[:, -1].permute(0, 2, 3, 1)[~mask], x[:, -1].permute(0, 2, 3, 1)[~mask]))
        self.assertTrue((valid[:, -1][mask]).all())
        model = self.model()
        seen = []
        valid_seen = []
        hook = model.encoder.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
        attn_hook = model.temporal.register_forward_pre_hook(lambda m, args: valid_seen.append(args[2].clone()))
        model(masked, valid, batch['dt'])
        hook.remove(); attn_hook.remove()
        self.assertTrue(torch.equal(seen[0], masked.flatten(0, 1)))
        self.assertTrue(valid_seen[0].any())
        dense = torch.ones(64, 512, dtype=torch.bool)
        spec = mask_settings({})
        a = patch_mask(dense, spec, torch.Generator().manual_seed(42))
        b = patch_mask(dense, spec, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertAlmostEqual(float(a.float().mean()), .5, places=5)
        ds = TemporalE2EMAEDataset([self.seq], self.cfg)
        first = ds[0]['mask']
        alternatives = []
        for epoch in range(1, 6):
            ds.set_epoch(epoch)
            alternatives.append(ds[0]['mask'])
        self.assertTrue(any(not torch.equal(first, value) for value in alternatives))
        val = TemporalE2EMAEDataset([self.seq], self.cfg, split='val')
        a = val[0]; val.set_epoch(99); b = val[0]
        self.assertTrue(torch.equal(a['mask'], b['mask']))
        self.assertTrue(torch.equal(a['x'], b['x']))

    def test_loss_values_exclusion_empty_nonfinite_and_gradients(self):
        pred = torch.zeros(1, 5, 1, 4, requires_grad=True)
        target = torch.zeros_like(pred).detach()
        target[:, :3, :, 0] = 1
        target[:, 3, :, 0] = 2
        target[:, 4, :, 0] = 3
        target[..., 1] = float('nan')
        target[..., 2] = 100
        valid = torch.tensor([[[True, False, True, True]]])
        mask = torch.tensor([[[True, True, False, False]]])
        loss, stats = reconstruction_loss(pred, target, valid, mask)
        self.assertAlmostEqual(float(loss.detach()), 2.375)
        self.assertEqual(float(stats['loss_xyz'].detach()), .5)
        self.assertEqual(float(stats['loss_range'].detach()), 1.5)
        self.assertEqual(float(stats['loss_intensity'].detach()), 2.5)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad[..., 1:] == 0).all())
        bad_pred = torch.full((1, 5, 1, 4), float('nan'), requires_grad=True)
        empty, _ = reconstruction_loss(bad_pred, target, valid, torch.zeros_like(mask))
        self.assertEqual(float(empty.detach()), 0.)
        empty.backward()
        self.assertTrue(torch.isfinite(bad_pred.grad).all())
        self.assertTrue((bad_pred.grad == 0).all())

    def trained_checkpoint(self):
        model, batch = self.model(), self.batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
        scheduler = mae.make_scheduler(optimizer, self.cfg['mae_train_params'])
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        names = ['encoder.down_cntx.conv1.weight', 'temporal.change.0.weight',
                 'decoder.up_block1.conv1.weight', 'decoder.up_block4.conv1.weight', 'decoder.logits.weight']
        before = {n: model.state_dict()[n].clone() for n in names}
        mae.phase(model, [batch], self.cfg, torch.device('cpu'), optimizer, scaler)
        for name in names:
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)
        self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(any(mos.gradient_health(model).values()))
        return model, mae.checkpoint_payload(model, optimizer, scheduler, scaler, self.cfg, 1, .9, [])

    def test_full_gradients_transfer_and_supervised_step(self):
        model, payload = self.trained_checkpoint()
        path = Path(self.tmp.name)/'mae.pt'
        torch.save(payload, path)
        cfg = deepcopy(self.mos_cfg)
        cfg['mos_model_params'] = {**self.cfg['mos_model_params'], 'num_classes':2}
        mos_model = mos.build_mos_model(cfg)
        head = {k: v.clone() for k,v in mos_model.state_dict().items() if k in HEAD}
        report = load_mae_weights(mos_model, path, cfg)
        self.assertEqual(set(report['excluded_keys']), HEAD)
        for key, val in model.state_dict().items():
            if key not in HEAD:
                self.assertTrue(torch.equal(val, mos_model.state_dict()[key]), key)
            else:
                self.assertTrue(torch.equal(head[key], mos_model.state_dict()[key]))
        b = self.batch()
        logits = mos_model(b['x'], b['valid'], b['dt'])
        self.assertEqual(logits.shape[1], 2)
        y = torch.randint(0, 2, (1, 16, 64))
        optimizer = torch.optim.AdamW(mos_model.parameters(), lr=.0005)
        loss = mos.make_criterion(cfg, torch.device('cpu'))(logits, y)
        loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_transfer_rejects_incomplete_or_incompatible_checkpoints(self):
        source, payload = self.trained_checkpoint()
        path = Path(self.tmp.name)/'mae.pt'
        cfg = deepcopy(self.cfg); cfg['mos_model_params']['num_classes'] = 2
        target = mos.build_mos_model(cfg)
        for kind in ('missing', 'unexpected', 'shape', 'normalization', 'channels', 'architecture', 'head', 'nan', 'format'):
            data = deepcopy(payload)
            if kind == 'missing': del data['model_state_dict']['temporal.gamma']
            if kind == 'unexpected': data['model_state_dict']['extra'] = torch.zeros(1)
            if kind == 'shape': data['model_state_dict']['encoder.down_cntx.conv1.weight'] = torch.zeros(1)
            if kind == 'normalization': data['config']['mos_data_params']['normalization'] = 'paper_normalization'
            if kind == 'channels': data['metadata']['channel_names'].reverse()
            if kind == 'architecture': data['config']['mos_model_params']['radius'] = 2
            if kind == 'head': data['model_state_dict']['decoder.logits.weight'] = torch.zeros(2,32,1,1)
            if kind == 'nan': data['model_state_dict']['temporal.gamma'].fill_(float('nan'))
            if kind == 'format': data['format'] = 'old-tmae'
            if kind in ('architecture', 'normalization'):
                data['metadata'] = mae.input_metadata(data['config'])
            torch.save(data, path)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_mae_weights(target, path, cfg)
        torch.save(payload, path)
        cfg['mos_data_params']['previous_offsets'] = [5, 1]
        with self.assertWarnsRegex(UserWarning, 'offsets differ'):
            load_mae_weights(target, path, cfg)

    def test_scheduler_and_configuration_baseline(self):
        _, base = mos.read_config(BASELINE)
        _, ft = mos.read_config(FINETUNE)
        ft['mos_log_params']['run_name'] = base['mos_log_params']['run_name']
        self.assertEqual(base, ft)
        _, config = mae.read_config(CONFIG)
        for key in ('model_params', 'mos_data_params', 'mos_augmentation'):
            self.assertEqual(config[key], base[key])
        self.assertEqual(config['mae_train_params']['epochs'], 50)
        self.assertEqual(config['mae_train_params']['amp_dtype'], 'bfloat16')
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.0005)
        train = config['mae_train_params']
        scheduler = mae.make_scheduler(optimizer, train)
        rates = []
        for epoch in range(50):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step(); scheduler.step()
        self.assertEqual(rates[:3], [.00025, .0005, .0005])
        self.assertAlmostEqual(rates[-1], .000005)
        for value in ({'patch_h':0}, {'mask_ratio':1.1}, {'apply_to':'history'}, {'mask_only_valid':'true'}):
            with self.assertRaises(ValueError): mask_settings(value)
        with self.assertRaises(ValueError): reconstruction_settings({'xyz_weight':float('inf')})

    def test_history_diagnostic(self):
        results = mae.history_diagnostics(self.model(), self.batch(), self.cfg, torch.device('cpu'))
        for mode in ('zero', 'copy_current'):
            self.assertTrue(np.isfinite(results[mode]['loss']))
            self.assertGreaterEqual(results[mode]['masked_prediction_mean_abs_delta'], 0)

    def test_legacy_tmae_four_channel_path(self):
        from models.salsanext_temporal_mae import SalsaNextTemporalMAE
        from loss_temporal_mae import temporal_mae_loss
        cfg = yaml.safe_load((ROOT/'src/configs/semanticKitti_tmae_rangeview.yaml').read_text())
        model = SalsaNextTemporalMAE(cfg)
        pair = torch.randn(1, 2, 5, 16, 32)
        mask = torch.zeros(1, 1, 16, 32)
        mask[:, :, :, :16] = 1
        pred = model(pair, mask)
        self.assertEqual(pred.shape, (1, 4, 16, 32))
        loss, _ = temporal_mae_loss(pred, pair[:, -1, :4], torch.zeros(1, 0, 16, 32),
                                    mask, torch.ones_like(mask), torch.ones_like(mask), cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_epoch_resume_reproduces_uninterrupted_run(self):
        # Capture an epoch boundary, then compare the next augmented epoch exactly.
        cfg = deepcopy(self.cfg)
        cfg['mos_augmentation']['enabled'] = True
        def setup():
            mos.seed_all(42)
            with patch.object(mos, 'make_sequences', return_value=[self.seq, {**self.seq, 'seq_id':'08'}]):
                loaders = [mae.make_loader(cfg, split, torch.device('cpu')) for split in ('train', 'val')]
            model = mos.build_mos_model(cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=.0005)
            return model, opt, mae.make_scheduler(opt, cfg['mae_train_params']), torch.amp.GradScaler('cuda', enabled=False), loaders
        model, opt, sched, scaler, loaders = setup()
        mae.set_epoch(loaders[0], 0)
        mae.phase(model, loaders[0], cfg, torch.device('cpu'), opt, scaler, max_batches=1)
        sched.step()
        state = mae.checkpoint_payload(model, opt, sched, scaler, cfg, 1, .9, [mae.capture_rng(loaders)])
        path = Path(self.tmp.name)/'resume.pt'; torch.save(state, path)
        mae.set_epoch(loaders[0], 1)
        expected = mae.phase(model, loaders[0], cfg, torch.device('cpu'), opt, scaler, max_batches=1)
        clone, opt2, sched2, scaler2, loaders2 = setup()
        epoch, best = mae.resume_training(path, clone, opt2, sched2, scaler2, cfg, loaders2, torch.device('cpu'))
        self.assertEqual((epoch, best), (1, .9))
        mae.set_epoch(loaders2[0], 1)
        actual = mae.phase(clone, loaders2[0], cfg, torch.device('cpu'), opt2, scaler2, max_batches=1)
        self.assertEqual(actual, expected)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, clone.state_dict()[k]), k)

    def test_trainer_checkpoint_logging_and_mos_finetuning_integration(self):
        cfg = deepcopy(self.cfg)
        cfg['mae_train_params']['epochs'] = 1
        config_path = Path(self.tmp.name)/'mae.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        args = SimpleNamespace(cfg_path=str(config_path), log_root=None, epochs=None, batch_size=None,
                               num_workers=None, max_batches=1, resume=None, dry_run=False, history_diagnostic=True)
        seqs = [self.seq, {**self.seq, 'seq_id':'08'}]
        with patch.object(mos, 'make_sequences', return_value=seqs):
            run = mae.train(args, torch.device('cpu'))
        self.assertTrue((run/'checkpoints/best_val_loss.pt').is_file())
        checkpoint = torch.load(run/'checkpoints/last.pt', weights_only=False)
        validate_mae_checkpoint(checkpoint)
        self.assertEqual(checkpoint['metadata']['channel_names'], ['x','y','z','range','intensity'])
        rows = (run/'csv/epochs.csv').read_text()
        for phase in ('train', 'val'):
            for key in ('loss_total', 'loss_xyz', 'loss_range', 'loss_intensity', 'valid_ratio', 'masked_valid_ratio'):
                self.assertIn(phase+'/'+key, rows)
        ft = deepcopy(self.mos_cfg)
        ft['mos_model_params'] = {**cfg['mos_model_params'], 'num_classes':2}
        ft['mos_train_params']['pretrained_path'] = str(run/'checkpoints/best_val_loss.pt')
        ft['mos_log_params'].update(log_root=str(Path(self.tmp.name)/'mos'),use_tensorboard=False)
        config_path = Path(self.tmp.name)/'mos.yaml'; config_path.write_text(yaml.safe_dump(ft))
        args.cfg_path = str(config_path); args.epochs = 1; args.batch_size=1; args.num_workers=0
        with patch.object(mos, 'make_sequences', return_value=seqs):
            mos.train(args, torch.device('cpu'))
        saved = next((Path(self.tmp.name)/'mos').glob('*/checkpoints/last.pt'))
        self.assertIn('pretraining_transfer', torch.load(saved, weights_only=False)['config'])

    def test_ddp_workers_two_training_steps(self):
        np.save(Path(self.tmp.name)/'seq.npy', self.seq)
        (Path(self.tmp.name)/'cfg.yaml').write_text(yaml.safe_dump(self.cfg))
        mp.spawn(ddp_check, args=(self.tmp.name,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
