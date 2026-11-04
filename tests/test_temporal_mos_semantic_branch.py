"""Prepared HPC checks for the current-bottleneck movable branch and gate.

NOT executed in the implementation environment. On a compute node:
MOS_CHECK_DEVICE=cuda MOS_CHECK_DDP=1 python3 -m unittest discover -s tests \
    -p 'test_temporal_mos_semantic_branch.py' -v
"""
from copy import deepcopy
import datetime
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from helper.mos_auxiliary import semantic_model_settings
from helper.temporal_e2e_transfer import checkpoint_format, input_metadata, load_mae_weights
from mos_models import build_mos_model
from mos_models.salsanext_temporal_e2e import SalsaNextTemporalE2E
import train_temporal_mos_e2e as trainer
from evaluate_mos_benchmark import load_adapter

REFERENCE = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml'
VARIANTS = {mode: ROOT/f'src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_{mode}.yaml'
            for mode in ('none', 'gate')}


def config(mode='none'):
    return yaml.safe_load(VARIANTS[mode].read_text())


def inputs(device):
    x = torch.randn(1, 2, 5, 32, 64, device=device)
    valid = torch.ones(1, 2, 32, 64, dtype=torch.bool, device=device)
    dt = torch.tensor([[-.8, 0.]], device=device)
    return x, valid, dt


def has_nonzero_gradient(module):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum() > 0)
               for p in module.parameters())


def semantic_ddp_worker(rank, directory):
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=120))
    try:
        for mode in ('none', 'gate'):
            trainer.seed_all(42 + rank, deterministic=True)
            cfg = config(mode)
            model = trainer.wrap_distributed(build_mos_model(cfg).to(device), device)
            criterion = trainer.make_criterion(cfg, device)
            optimizer = trainer.make_optimizer(model.parameters(), cfg['mos_train_params'])
            assert {id(p) for p in model.parameters() if p.requires_grad} == {
                id(p) for group in optimizer.param_groups for p in group['params']}
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                mos, movable = model(*inputs(device), return_aux=True)
                target = torch.randint(0, 2, (1, 32, 64), device=device)
                aux_target = torch.randint(0, 2, target.shape, device=device)
                if step == 1 and rank == 0:
                    aux_target.fill_(-1)  # local empty supervision still joins both task reductions
                mos_parts = criterion.components(mos, target)
                aux_parts = criterion.movable_criterion.components(movable, aux_target)
                loss = trainer.corrected_task_loss(criterion, mos_parts, target)
                loss = loss + criterion.auxiliary_loss_weight * trainer.corrected_task_loss(
                    criterion.movable_criterion, aux_parts, aux_target)
                loss.backward()
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None, name
                    assert torch.isfinite(parameter.grad).all(), name
                assert has_nonzero_gradient(model.module.semantic_adapter)
                assert has_nonzero_gradient(model.module.movable_head)
                if mode == 'gate':
                    assert has_nonzero_gradient(model.module.semantic_gate)
                optimizer.step()
                print(f'[semantic-ddp-check] mode={mode} rank={rank} world_size=2 step={step+1} loss={float(loss.detach()):.6f}', flush=True)
            del optimizer, model
    finally:
        dist.destroy_process_group()


class SemanticBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device = torch.device(os.environ.get('MOS_CHECK_DEVICE', 'cpu'))
        if cls.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('MOS_CHECK_DEVICE=cuda requested but CUDA is unavailable')

    def setUp(self):
        trainer.seed_all(42, deterministic=True)

    def model(self, mode='none'):
        return build_mos_model(config(mode)).to(self.device).eval()

    def test_exact_yaml_inheritance_and_single_ablation_switch(self):
        reference = yaml.safe_load(REFERENCE.read_text())
        first, second = config('none'), config('gate')
        for cfg in (first, second):
            reduced = deepcopy(cfg)
            reduced['mos_model_params'].pop('movable_branch')
            reduced['mos_model_params'].pop('semantic_fusion')
            reduced['mos_log_params']['run_name'] = reference['mos_log_params']['run_name']
            self.assertEqual(reduced, reference)
            self.assertIsNone(cfg['mos_train_params']['pretrained_path'])
        second['mos_model_params']['semantic_fusion'] = first['mos_model_params']['semantic_fusion']
        second['mos_log_params']['run_name'] = first['mos_log_params']['run_name']
        self.assertEqual(first, second)
        accumulation = int(reference['mos_train_params']['gradient_accumulation'])
        self.assertGreaterEqual(2 // accumulation, 2)  # documented --max_batches 2

    def test_legacy_defaults_keys_rng_and_strict_loading(self):
        legacy_cfg = yaml.safe_load(REFERENCE.read_text())
        trainer.seed_all(42)
        implicit = build_mos_model(legacy_cfg).to(self.device).eval()
        rng_after = torch.get_rng_state()
        explicit_cfg = deepcopy(legacy_cfg)
        explicit_cfg['mos_model_params'].update(movable_branch='shared_decoder', semantic_fusion='none')
        trainer.seed_all(42)
        explicit = build_mos_model(explicit_cfg).to(self.device).eval()
        self.assertTrue(torch.equal(rng_after, torch.get_rng_state()))
        self.assertFalse(hasattr(implicit, 'semantic_adapter'))
        self.assertFalse(hasattr(implicit, 'semantic_gate'))
        self.assertEqual({k for k in implicit.state_dict() if k.startswith('movable_head.')},
                         {'movable_head.weight', 'movable_head.bias'})
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        trainer.validate_resume_model_params(legacy_cfg, explicit_cfg)
        trainer.validate_resume_model_params(explicit_cfg, legacy_cfg)
        with torch.no_grad():
            values = inputs(self.device)
            torch.testing.assert_close(implicit(*values), explicit(*values), rtol=0, atol=0)
        # Compare against direct legacy construction, not only the builder.
        trainer.seed_all(42)
        original = SalsaNextTemporalE2E(in_channels=5, auxiliary_movable=True).to(self.device)
        for key, tensor in implicit.state_dict().items():
            torch.testing.assert_close(tensor, original.state_dict()[key], rtol=0, atol=0)
        mos_cfg = deepcopy(legacy_cfg); mos_cfg.pop('mos_auxiliary')
        mos_only = build_mos_model(mos_cfg).to(self.device).eval()
        old_mos = SalsaNextTemporalE2E(in_channels=5).to(self.device).eval()
        mos_only.load_state_dict(old_mos.state_dict(), strict=True)
        self.assertFalse(hasattr(mos_only, 'movable_head'))
        self.assertFalse(hasattr(mos_only, 'semantic_adapter'))
        self.assertFalse(hasattr(mos_only, 'semantic_gate'))
        with torch.no_grad():
            torch.testing.assert_close(mos_only(*values), old_mos(*values), rtol=0, atol=0)

    def test_invalid_modes_fail_in_builder_trainer_and_benchmark(self):
        invalid = []
        for changes in (dict(movable_branch='wrong'), dict(semantic_fusion='wrong'),
                        dict(movable_branch='shared_decoder', semantic_fusion='bottleneck_gate'),
                        dict(name='unet_small'), dict(num_classes=5), dict(fusion='wrong')):
            cfg = config(); cfg['mos_model_params'].update(changes); invalid.append(cfg)
        cfg = config(); cfg['mos_auxiliary']['enabled'] = False; invalid.append(cfg)
        cfg = config(); cfg['mos_auxiliary'] = dict(enabled=True, task='residual'); invalid.append(cfg)
        with tempfile.TemporaryDirectory() as directory:
            for index, cfg in enumerate(invalid):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError): build_mos_model(cfg)
                    path = Path(directory)/f'bad{index}.yaml'
                    path.write_text(yaml.safe_dump(cfg))
                    with self.assertRaises(ValueError): trainer.read_config(path)
            # The benchmark must reject invalid saved options before state loading.
            bad = config(); bad['mos_model_params']['semantic_fusion'] = 'wrong'
            path = Path(directory)/'bad.pt'
            torch.save(dict(config=bad, model_state_dict={}), path)
            with self.assertRaisesRegex(ValueError, 'semantic_fusion'): load_adapter(path, self.device)

    def test_initialization_matches_between_ablations(self):
        trainer.seed_all(42)
        first = self.model('none'); first_rng = torch.get_rng_state()
        trainer.seed_all(42)
        second = self.model('gate')
        self.assertTrue(torch.equal(first_rng, torch.get_rng_state()))
        self.assertEqual(set(second.state_dict()) - set(first.state_dict()),
                         {'semantic_gate.weight', 'semantic_gate.bias'})
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertTrue((second.semantic_gate.weight == 0).all())
        self.assertTrue((second.semantic_gate.bias == 0).all())
        values = inputs(self.device)
        with torch.no_grad(): torch.testing.assert_close(first(*values), second(*values), rtol=0, atol=0)

    def test_shapes_quarter_skip_single_encoder_and_decoder(self):
        model = self.model(); observed = {}; calls = []
        def encoder_hook(module, args, output):
            observed['encoder'] = output; calls.append('encoder')
        def skip_hook(module, args): observed['skip'] = args[0]
        def logits_hook(module, args, output): observed['quarter_logits'] = output
        def decoder_hook(module, args, output): calls.append('decoder')
        hooks = [model.encoder.register_forward_hook(encoder_hook),
                 model.movable_head.skip_projection.register_forward_pre_hook(skip_hook),
                 model.movable_head.logits.register_forward_hook(logits_hook),
                 model.decoder.register_forward_hook(decoder_hook)]
        try:
            with torch.no_grad(): mos, movable = model(*inputs(self.device), return_aux=True)
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(calls, ['encoder', 'decoder'])
        self.assertEqual(observed['encoder'][0].shape, (2, 256, 2, 4))
        self.assertEqual(observed['skip'].shape, (1, 256, 8, 16))
        self.assertEqual(observed['quarter_logits'].shape, (1, 2, 8, 16))
        self.assertEqual(mos.shape, (1, 2, 32, 64)); self.assertEqual(movable.shape, mos.shape)

    def test_return_aux_diagnostics_and_lazy_movable_decoder(self):
        for mode in ('none', 'gate'):
            with self.subTest(mode=mode):
                model = self.model(mode); values = inputs(self.device)
                with torch.no_grad():
                    plain = model(*values)
                    (mos, movable), diagnostics = model(*values, return_aux=True, return_diagnostics=True)
                    torch.testing.assert_close(plain, mos, rtol=0, atol=0)
                    self.assertEqual(movable.shape, mos.shape); self.assertIsInstance(diagnostics, dict)
                    with patch.object(model.movable_head, 'forward', side_effect=AssertionError('head should be lazy')):
                        torch.testing.assert_close(model(*values), plain, rtol=0, atol=0)
                    if mode == 'none':
                        with patch.object(model.semantic_adapter, 'forward', side_effect=AssertionError('adapter should be lazy')):
                            torch.testing.assert_close(model(*values), plain, rtol=0, atol=0)

    def test_without_gate_semantic_weights_have_no_direct_mos_effect(self):
        model = self.model(); values = inputs(self.device)
        with torch.no_grad():
            original = model(*values)
            for module in (model.semantic_adapter, model.movable_head):
                for parameter in module.parameters(): parameter.add_(1.)
            mos, _ = model(*values, return_aux=True)
        torch.testing.assert_close(original, mos, rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
        model(*values).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.semantic_adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in model.movable_head.parameters()))

    def test_shared_semantic_tensor_and_current_only_movable(self):
        model = self.model('gate'); values = inputs(self.device); seen = []
        a = model.movable_head.register_forward_pre_hook(lambda module, args: seen.append(('head', args[0])))
        b = model.semantic_gate.register_forward_pre_hook(lambda module, args: seen.append(('gate', args[0])))
        try:
            with torch.no_grad(): _, original = model(*values, return_aux=True)
        finally:
            a.remove(); b.remove()
        self.assertEqual({name for name, _ in seen}, {'head', 'gate'})
        self.assertIs(seen[0][1], seen[1][1])
        changed = values[0].clone(); changed[:, 0] += 10.
        with torch.no_grad(): _, altered = model(changed, values[1], values[2], return_aux=True)
        torch.testing.assert_close(original, altered, rtol=0, atol=0)

    def test_gate_first_gradient_and_mos_gradient_into_semantics_after_update(self):
        model = self.model('gate'); values = inputs(self.device)
        target = torch.randint(0, 2, (1, 32, 64), device=self.device)
        seen = []
        def retain(module, args, output): output.retain_grad(); seen.append(output)
        hook = model.semantic_adapter.register_forward_hook(retain)
        try:
            F.cross_entropy(model(*values), target).backward()
            self.assertTrue(has_nonzero_gradient(model.semantic_gate))
            self.assertIsNotNone(seen[-1].grad)
            self.assertEqual(float(seen[-1].grad.abs().sum()), 0.)  # exact zero P
            optimizer = torch.optim.SGD(model.semantic_gate.parameters(), lr=.1)
            optimizer.step()
            # Deliberately nontrivial P makes the inference-effect check robust
            # even if the first optimizer update is smaller than FP32 rounding.
            with torch.no_grad():
                model.semantic_gate.weight.add_(.01); model.semantic_gate.bias.add_(.1)
            model.zero_grad(set_to_none=True)
            changed = model(*values)
            F.cross_entropy(changed, target).backward()
            self.assertTrue(has_nonzero_gradient(model.semantic_adapter))
            self.assertGreater(float(seen[-1].grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.movable_head.parameters()))
        finally:
            hook.remove()
        with torch.no_grad():
            model.semantic_gate.weight.zero_(); model.semantic_gate.bias.zero_()
            identity = model(*values)
        self.assertGreater(float((identity-changed.detach()).abs().max()), 0.)

    def test_gate_added_term_respects_current_bottleneck_mask(self):
        model = self.model('gate'); values = inputs(self.device)
        with torch.no_grad(): model.semantic_gate.bias.fill_(.8)
        valid = values[1].clone(); valid[:, -1] = False
        seen = {}
        a = model.temporal.register_forward_hook(lambda module, args, output: seen.update(before=output))
        b = model.decoder.register_forward_pre_hook(lambda module, args: seen.update(after=args[0]))
        try:
            with torch.no_grad(): model(values[0], valid, values[2])
        finally:
            a.remove(); b.remove()
        torch.testing.assert_close(seen['before'], seen['after'], rtol=0, atol=0)

    def test_movable_loss_and_total_loss_train_active_parameters(self):
        for mode in ('none', 'gate'):
            with self.subTest(mode=mode):
                model = self.model(mode); cfg = config(mode)
                criterion = trainer.make_criterion(cfg, self.device)
                target = torch.randint(0, 2, (1, 32, 64), device=self.device)
                values = inputs(self.device)
                _, movable = model(*values, return_aux=True)
                criterion.movable_criterion(movable, target).backward()
                self.assertTrue(has_nonzero_gradient(model.semantic_adapter))
                self.assertTrue(has_nonzero_gradient(model.movable_head))
                self.assertTrue(has_nonzero_gradient(model.encoder))
                self.assertIsNone(model.decoder.logits.weight.grad)
                model.zero_grad(set_to_none=True)
                mos, movable = model(*values, return_aux=True)
                total = criterion(mos, target) + criterion.auxiliary_loss_weight * criterion.movable_criterion(movable, target)
                self.assertTrue(torch.isfinite(total)); total.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer = trainer.make_optimizer(model.parameters(), cfg['mos_train_params'])
                self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                                 {id(p) for group in optimizer.param_groups for p in group['params']})

    def test_resume_rejects_other_ablation_and_strict_checkpoint_roundtrip(self):
        with self.assertRaisesRegex(ValueError, 'semantic_fusion'):
            trainer.validate_resume_model_params(config('none'), config('gate'))
        for mode in ('none', 'gate'):
            cfg = config(mode); model = self.model(mode); values = inputs(self.device)
            metadata = semantic_model_settings(cfg)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'checkpoint.pt'
                torch.save(dict(config=cfg, model_state_dict=model.state_dict(), model_options=metadata), path)
                # This is the actual benchmark's saved-config construction path.
                loaded, saved_cfg, payload = load_adapter(path, self.device)
                self.assertEqual(payload['model_options'], metadata)
                self.assertEqual(semantic_model_settings(saved_cfg), metadata)
                with torch.no_grad():
                    torch.testing.assert_close(model(*values), loaded(*values), rtol=0, atol=0)
                other = self.model('gate' if mode == 'none' else 'none')
                with self.assertRaises(RuntimeError): other.load_state_dict(model.state_dict(), strict=True)

    def test_mae_transfer_fresh_semantic_modules_and_strict_backbone(self):
        source_cfg = yaml.safe_load(REFERENCE.read_text())
        source_cfg.pop('mos_auxiliary'); source_cfg['mos_model_params']['num_classes'] = 5
        source = build_mos_model(source_cfg)
        payload = dict(format=checkpoint_format(source_cfg), config=source_cfg,
                       metadata=input_metadata(source_cfg), epoch=1, model_state_dict=source.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mae.pt'; torch.save(payload, path)
            for mode in ('none', 'gate'):
                cfg = config(mode); target = build_mos_model(cfg)
                fresh = {key: value.clone() for key, value in target.state_dict().items()
                         if key.startswith(('decoder.logits.', 'movable_head.', 'semantic_adapter.', 'semantic_gate.'))}
                report = load_mae_weights(target, path, cfg)
                self.assertEqual(set(report['excluded_keys']), set(fresh))
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, fresh[key] if key in fresh else source.state_dict()[key])
            broken = deepcopy(payload)
            broken['model_state_dict'].pop(next(k for k in source.state_dict() if k.startswith('encoder.')))
            torch.save(broken, path)
            with self.assertRaisesRegex(ValueError, 'key mismatch'):
                load_mae_weights(build_mos_model(config()), path, config())

    @unittest.skipUnless(os.environ.get('MOS_CHECK_DDP') == '1', 'enable MOS_CHECK_DDP=1 on a two-GPU compute node')
    def test_two_gpu_ddp_both_ablations_two_updates(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory(prefix='semantic-ddp-check-') as directory:
            mp.spawn(semantic_ddp_worker, args=(directory,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
