"""Inference interventions, unchanged checkpoints, full KNN benchmark and comparison."""
from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from mos_models.salsanext_temporal_e2e import SalsaNextTemporalE2E
from evaluate_mos_benchmark import (main, load_adapter, apply_temporal_ablation,
    temporal_ablation_metadata, temporal_gamma_metadata, semantic_gate_ablation_metadata)
from compare_temporal_ablation import compare, MODES, PROTOCOL_KEYS
from test_semantic_gate_eval_ablation import config


def dataset(root, count=10):
    seq = root/'dataset/sequences/08'
    (seq/'velodyne').mkdir(parents=True)
    (seq/'labels').mkdir()
    for index in range(count):
        np.array([[4+index*.1, 0, -.1, .1], [8, 0, -.2, .2],
                  [0, 5, -.1, .3], [0, -7, -.4, .2]], np.float32).tofile(seq/'velodyne'/f'{index:06d}.bin')
        np.array([252, 10, 40, 0], '<u4').tofile(seq/'labels'/f'{index:06d}.label')
    identity = '1 0 0 0 0 1 0 0 0 0 1 0'
    (seq/'poses.txt').write_text((identity+'\n')*count)
    (seq/'calib.txt').write_text('Tr: '+identity+'\n')


def write_comparison_fixture(root):
    common = dict(checkpoint='/same.pt', checkpoint_sha256='same-sha', checkpoint_epoch=85,
                  checkpoint_model_config=config(), eval_sequences=['08'], dataset_root='/same-data',
                  n_scans=1, moving_threshold=.5, projection={'height':32,'width':64},
                  normalization='paper_normalization', input_mode='range_xyz_intensity',
                  history_init='zero_missing', history_reference_offsets=[8], pose_source='same',
                  gt_mapping='same', postprocessing=dict(method='knn',knn=5,search=5,sigma=1.,cutoff=1.),
                  partial_smoke_test=True, pipeline_git_commit='same',
                  threshold_sweep={'enabled':False}, movable_diagnostics={'enabled':False})
    for mode, counts in zip(MODES, ((8,2,2,8), (6,4,4,6), (4,6,6,4))):
        folder = root/mode
        folder.mkdir()
        meta = deepcopy(common)
        meta['temporal_ablation'] = dict(mode=mode, checkpoint_unchanged=True,
            historical_inputs_masked=mode == 'mask_history', temporal_module_executed=mode != 'bypass_temporal',
            semantic_gate_unchanged=True)
        meta['semantic_gate_ablation'] = dict(checkpoint_fusion='bottleneck_gate', gate_enabled=True,
                                            disable_semantic_gate=False, checkpoint_unchanged=True,
                                            supported=True, mode='runtime_forward_bypass')
        (folder/'benchmark_eval_config.yaml').write_text(yaml.safe_dump(meta))
        (folder/'projection_coverage.csv').write_text('sequence,frame,n_points,history_available\n08,000008,20,[True]\n')
        # Stored IoUs are deliberately false: only counts may influence comparison.
        (folder/'point_metrics.csv').write_text('tp_moving,fp_moving,fn_moving,tn_static,n_gt_valid,n_points,n_scans,point_moving_iou\n'
            + ','.join(map(str, (*counts,20,20,1,.123456)))+'\n')
        (folder/'pixel_metrics.csv').write_text('sequence,tp_moving,fp_moving,fn_moving,tn_static,valid_pixels,correct,n_scans,pixel_moving_iou\n'
            + ','.join(map(str, ('ALL',*counts,20,counts[0]+counts[3],1,.987654)))+'\n')


class TemporalHistoryAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(13)
        self.model = SalsaNextTemporalE2E(in_channels=5, dropout=0., auxiliary_task='movable',
            movable_branch='current_bottleneck', semantic_fusion='bottleneck_gate').eval()
        self.inputs = (torch.randn(1,3,5,32,64), torch.ones(1,3,32,64,dtype=torch.bool),
                       torch.tensor([[-1.,-.8,0.]]))

    def test_none_default_and_explicit_are_bitwise_identical_without_rng(self):
        state = {k:v.clone() for k,v in self.model.state_dict().items()}
        before = torch.get_rng_state().clone()
        with torch.inference_mode():
            normal = self.model(*self.inputs)
            explicit = self.model(*apply_temporal_ablation(self.inputs,'none'), disable_temporal_fusion=False)
        self.assertIs(apply_temporal_ablation(self.inputs), self.inputs)
        self.assertTrue(torch.equal(normal,explicit))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v,state[k]),k)

    def test_mask_copies_all_history_preserves_current_dt_and_no_match_path(self):
        original = tuple(x.clone() for x in self.inputs)
        masked = apply_temporal_ablation(self.inputs,'mask_history')
        self.assertFalse(masked[0][:,:-1].any())
        self.assertFalse(masked[1][:,:-1].any())
        self.assertTrue(torch.equal(masked[0][:,-1],original[0][:,-1]))
        self.assertTrue(torch.equal(masked[1][:,-1],original[1][:,-1]))
        self.assertTrue(torch.equal(masked[2],original[2]))
        for a,b in zip(self.inputs,original):
            self.assertTrue(torch.equal(a,b))
        for a,b in zip(masked,self.inputs):
            self.assertEqual(a.shape,b.shape)
        with patch.object(self.model.temporal,'forward',wraps=self.model.temporal.forward) as attention, torch.inference_mode():
            logits, diagnostics = self.model(*masked,return_diagnostics=True)
        self.assertEqual(attention.call_count,1)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(diagnostics['null_weight'],torch.ones_like(diagnostics['null_weight']),rtol=0,atol=0)
        self.assertFalse(diagnostics['best_match'].any())
        # A learned no-match value can still change the fused result.
        with torch.no_grad():
            self.model.temporal.gamma.fill_(1.)
            self.model.temporal.null_value.fill_(100.)
        with torch.inference_mode():
            changed = self.model(*masked)
        self.assertFalse(torch.equal(changed,logits))

    def test_bypass_exact_current_bottleneck_gate_skips_and_aux(self):
        with torch.no_grad():
            self.model.semantic_gate.weight.fill_(.02)
            self.model.semantic_gate.bias.fill_(.7)
        encoder, gate_inputs, gate_outputs, decoder = [],[],[],[]
        hooks = [self.model.encoder.register_forward_hook(lambda m,a,o: encoder.append(o)),
                 self.model.semantic_adapter.register_forward_pre_hook(lambda m,a: gate_inputs.append(a[0].clone())),
                 self.model.semantic_gate.register_forward_hook(lambda m,a,o: gate_outputs.append(o.clone())),
                 self.model.decoder.register_forward_pre_hook(lambda m,a: decoder.append(a))]
        try:
            with patch.object(self.model.temporal,'forward',side_effect=AssertionError('Attention must be skipped')), torch.inference_mode():
                logits, aux = self.model(*self.inputs,disable_temporal_fusion=True,return_aux=True)
                off = self.model(*self.inputs,disable_temporal_fusion=True,disable_semantic_gate=True)
        finally:
            for hook in hooks:
                hook.remove()
        encoded, skips = encoder[0]
        current = encoded.reshape(1,3,*encoded.shape[1:])[:,-1]
        self.assertTrue(torch.equal(current,gate_inputs[0]))
        self.assertEqual(len(gate_outputs),1)
        torch.testing.assert_close(decoder[0][0],current+.1*current*torch.tanh(gate_outputs[0]),rtol=0,atol=0)
        self.assertTrue(torch.equal(decoder[1][0],current))
        for actual,skip in zip(decoder[0][1],skips):
            self.assertTrue(torch.equal(actual,skip.reshape(1,3,*skip.shape[1:])[:,-1]))
        self.assertEqual(logits.shape,(1,2,32,64))
        self.assertEqual(aux.shape,logits.shape)
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(aux).all())
        self.assertFalse(torch.equal(logits,off))
        with self.assertRaisesRegex(ValueError,'diagnostics'):
            self.model(*self.inputs,disable_temporal_fusion=True,return_diagnostics=True)

    def test_strict_checkpoint_parameter_keys_and_forward_options_unchanged(self):
        state = {k:v.clone() for k,v in self.model.state_dict().items()}
        keys = set(state)
        parameter_count = sum(p.numel() for p in self.model.parameters())
        with torch.inference_mode():
            for mode in MODES:
                self.model(*apply_temporal_ablation(self.inputs,mode),return_aux=True,
                           disable_temporal_fusion=mode=='bypass_temporal')
        self.assertEqual(keys,set(self.model.state_dict()))
        self.assertEqual(parameter_count,sum(p.numel() for p in self.model.parameters()))
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v,state[k]),k)
        self.model.load_state_dict(state,strict=True)
        with torch.inference_mode():
            output,diagnostics = self.model(*self.inputs,return_aux=True,return_diagnostics=True,disable_semantic_gate=True)
        self.assertEqual(len(output),2)
        self.assertIn('null_weight',diagnostics)

    def test_metadata_gamma_unsupported_models_and_invalid_cli_combinations(self):
        gamma = self.model.temporal.gamma.detach().clone()
        stats = temporal_gamma_metadata(self.model)
        self.assertAlmostEqual(stats['gamma_mean_abs'],float(gamma.abs().mean()))
        self.assertTrue(torch.equal(gamma,self.model.temporal.gamma))
        for mode in MODES:
            meta = temporal_ablation_metadata(self.model,mode)
            self.assertEqual(meta['mode'],mode)
            self.assertEqual(meta['temporal_module_executed'],mode!='bypass_temporal')
            self.assertTrue(meta['semantic_gate_unchanged'])
        for mode in MODES[1:]:
            with self.assertRaisesRegex(ValueError,'SalsaNextTemporalE2E'):
                temporal_ablation_metadata(torch.nn.Identity(),mode)
            with self.assertRaisesRegex(ValueError,'semantic-gate'):
                temporal_ablation_metadata(self.model,mode,True)
        args = ['--checkpoint','unused.pt','--dataset-root','unused','--output-dir','unused','--temporal-ablation','mask_history']
        for flags in (['--disable-semantic-gate'],['--threshold-sweep'],['--moving-threshold','.7'],['--movable-diagnostics']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                main(args+flags)
        with self.assertRaisesRegex(ValueError,'Unknown'):
            apply_temporal_ablation(self.inputs,'bad')

    def test_full_synthetic_knn_benchmark_all_modes_unchanged_gt_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            dataset(root)
            checkpoint=root/'checkpoint.pt'
            torch.save(dict(config=config(),epoch=85,model_state_dict=self.model.state_dict()),checkpoint)
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            labels={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'dataset/sequences/08/labels').glob('*.label')}
            common=['--checkpoint',str(checkpoint),'--dataset-root',str(root/'dataset'),'--device','cpu',
                    '--mode','both','--postprocess','knn','--knn-k','5','--knn-search','5',
                    '--knn-sigma','1.0','--knn-cutoff','1.0','--moving-threshold','0.5']
            for mode in MODES:
                main(common+['--temporal-ablation',mode,'--output-dir',str(root/mode)])
                meta=yaml.safe_load((root/mode/'benchmark_eval_config.yaml').read_text())
                self.assertEqual(meta['temporal_ablation'],temporal_ablation_metadata(self.model,mode))
                self.assertEqual(meta['n_scans'],10)
                self.assertEqual(meta['history_reference_offsets'],[8])
                self.assertEqual(meta['checkpoint_sha256'],digest)
                self.assertEqual(meta['temporal_attention_gamma'],temporal_gamma_metadata(self.model))
                self.assertTrue((root/mode/'point_metrics.csv').is_file())
                self.assertTrue((root/mode/'pixel_metrics.csv').is_file())
                coverage=list(csv.DictReader((root/mode/'projection_coverage.csv').read_text().splitlines()))
                self.assertEqual(coverage[-1]['history_available'],'[True]')
                self.assertEqual(coverage[-1]['history_used'],'[True]' if mode=='none' else '[False]')
            main(common+['--output-dir',str(root/'default')])
            for name in ('point_metrics.csv','pixel_metrics.csv'):
                self.assertEqual((root/'default'/name).read_bytes(),(root/'none'/name).read_bytes())
            rows=compare(*(root/mode for mode in MODES),root/'comparison')
            self.assertTrue(any(row['metric']=='pixel_moving_iou' for row in rows))
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            for p in (root/'dataset/sequences/08/labels').glob('*.label'):
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),labels[p.name])
            # Strictly reload the same old-format checkpoint after every mode.
            model,_,_=load_adapter(checkpoint,torch.device('cpu'))
            self.assertEqual(set(model.state_dict()),set(self.model.state_dict()))
            with self.assertRaises(FileExistsError):
                main(common+['--output-dir',str(root/'none')])

    def test_comparison_recomputes_counts_deltas_in_percentage_points(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); write_comparison_fixture(root)
            rows=compare(*(root/m for m in MODES),root/'comparison')
            by_key={row['metric']:row for row in rows}
            for level in ('point','pixel'):
                row=by_key[f'{level}_moving_iou']
                self.assertAlmostEqual(row['normal'],8/12)
                self.assertAlmostEqual(row['delta_mask_history_minus_normal'],6/14-8/12)
                self.assertAlmostEqual(row['delta_bypass_temporal_percentage_points'],100*(4/16-8/12))
                self.assertEqual(by_key[f'{level}_tp_moving']['normal'],8)
            self.assertTrue((root/'comparison/temporal_ablation_comparison.csv').is_file())
            self.assertTrue((root/'comparison/temporal_ablation_comparison.yaml').is_file())

    def test_comparison_rejects_incompatible_protocol_modes_scans_and_gt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); write_comparison_fixture(root)
            p=root/'mask_history/benchmark_eval_config.yaml'
            original=yaml.safe_load(p.read_text())
            changes={key:'different' for key in PROTOCOL_KEYS}
            changes.update(temporal_ablation={'mode':'none'},semantic_gate_ablation={'disable_semantic_gate':True},
                           threshold_sweep={'enabled':True},movable_diagnostics={'enabled':True})
            for key,value in changes.items():
                modified=deepcopy(original); modified[key]=value
                p.write_text(yaml.safe_dump(modified))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    compare(*(root/m for m in MODES),root/'bad')
            p.write_text(yaml.safe_dump(original))
            coverage=root/'mask_history/projection_coverage.csv'; saved=coverage.read_text()
            for changed in (saved.replace('000008','000009'),saved.replace(',20,',',21,')):
                coverage.write_text(changed)
                with self.assertRaisesRegex(ValueError,'scan identities'):
                    compare(*(root/m for m in MODES),root/'bad')
            coverage.write_text(saved)
            counts=root/'mask_history/point_metrics.csv'; saved=counts.read_text()
            # Same total but a different number of GT moving points is rejected.
            counts.write_text(saved.replace('6,4,4,6,20','7,3,4,6,20'))
            with self.assertRaisesRegex(ValueError,'ground-truth'):
                compare(*(root/m for m in MODES),root/'bad')
            counts.write_text(saved)
            (root/'mask_history/pixel_metrics.csv').unlink()
            with self.assertRaisesRegex(ValueError,'Pixel metrics'):
                compare(*(root/m for m in MODES),root/'bad')


if __name__=='__main__':
    unittest.main()
