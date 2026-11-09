"""Runtime-only semantic gate bypass and paired benchmark regression tests."""
from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from mos_models.salsanext_temporal_e2e import SalsaNextTemporalE2E
from evaluate_mos_benchmark import main, load_adapter, semantic_gate_ablation_metadata
from compare_semantic_gate_ablation import compare


def config():
    return dict(model_params=dict(grid_height=32, grid_width=64, FOV_UP=3., FOV_DOWN=-25., min_range=0., max_range=None),
                mos_data_params=dict(input_mode='range_xyz_intensity', normalization='paper_normalization',
                                     previous_offsets=[8], scan_rate_hz=10., projection_mode='lmnet', alignment_version='lmnet_v1'),
                mos_model_params=dict(name='salsanext_temporal_e2e', in_channels=5, num_classes=2,
                                      dropout=0., movable_branch='current_bottleneck', semantic_fusion='bottleneck_gate'),
                mos_auxiliary=dict(enabled=True, task='movable'))


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(13)
        self.model = SalsaNextTemporalE2E(in_channels=5, dropout=0., auxiliary_task='movable',
                                         movable_branch='current_bottleneck', semantic_fusion='bottleneck_gate').eval()
        self.inputs = (torch.randn(1, 2, 5, 32, 64), torch.ones(1, 2, 32, 64, dtype=torch.bool), torch.tensor([[-.8, 0.]]))

    def test_default_false_and_zero_gate_exact_identity(self):
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with torch.inference_mode():
            default = self.model(*self.inputs)
            explicit = self.model(*self.inputs, disable_semantic_gate=False)
            off = self.model(*self.inputs, disable_semantic_gate=True)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(default, off))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]), k)
        self.model.load_state_dict(state, strict=True)

    def test_learned_gate_changes_only_formula_and_aux_remains_independent(self):
        with torch.no_grad():
            self.model.semantic_gate.weight.fill_(.02)
            self.model.semantic_gate.bias.fill_(.7)
        temporal, decoder, gate = [], [], []
        hooks = [self.model.temporal.register_forward_hook(lambda m, a, o: temporal.append((o[0] if isinstance(o, tuple) else o).clone())),
                 self.model.decoder.register_forward_pre_hook(lambda m, a: decoder.append(a[0].clone())),
                 self.model.semantic_gate.register_forward_hook(lambda m, a, o: gate.append(o.clone()))]
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.inference_mode():
            on, aux_on = self.model(*self.inputs, return_aux=True)
            off, aux_off = self.model(*self.inputs, return_aux=True, disable_semantic_gate=True)
            plain = self.model(*self.inputs, disable_semantic_gate=True)
            output, diagnostics = self.model(*self.inputs, disable_semantic_gate=True, return_aux=True, return_diagnostics=True)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(gate), 1)
        torch.testing.assert_close(decoder[0], temporal[0] + .1*temporal[0]*torch.tanh(gate[0]), rtol=0, atol=0)
        self.assertTrue(torch.equal(decoder[1], temporal[1]))
        self.assertTrue(torch.equal(temporal[0], temporal[1]))
        self.assertFalse(torch.equal(on, off))
        self.assertTrue(torch.equal(aux_on, aux_off))
        self.assertTrue(torch.equal(plain, off))
        self.assertTrue(torch.equal(output[0], off))
        self.assertEqual(off.shape, (1, 2, 32, 64))
        self.assertTrue(torch.isfinite(off).all())
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]), k)

    def test_off_skips_semantic_and_movable_computation(self):
        from unittest.mock import patch
        with patch.object(self.model.semantic_adapter, 'forward', side_effect=AssertionError('adapter called')), \
             patch.object(self.model.movable_head, 'forward', side_effect=AssertionError('head called')), \
             torch.inference_mode():
            self.model(*self.inputs, disable_semantic_gate=True)

    def test_incompatible_models_and_strict_missing_gate(self):
        for task, branch in ((None, 'shared_decoder'), ('movable', 'shared_decoder'), ('movable', 'current_bottleneck'), ('residual', 'shared_decoder')):
            model = SalsaNextTemporalE2E(in_channels=5, auxiliary_task=task, movable_branch=branch).eval()
            with self.assertRaisesRegex(ValueError, 'bottleneck_gate'):
                semantic_gate_ablation_metadata(model, True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            state = self.model.state_dict()
            del state['semantic_gate.weight']
            torch.save(dict(cfg=config(), model_state_dict=state), path)
            with self.assertRaisesRegex(RuntimeError, 'semantic_gate.weight'):
                load_adapter(path, torch.device('cpu'))

    def test_nonfinite_gate_parameters_rejected(self):
        with torch.no_grad():
            self.model.semantic_gate.bias[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            semantic_gate_ablation_metadata(self.model, True)

    def test_synthetic_knn_benchmark_on_off_roundtrip_and_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            seq = root/'dataset/sequences/08'
            (seq/'velodyne').mkdir(parents=True)
            (seq/'labels').mkdir()
            for index in range(2):
                np.array([[4, 0, 0, .1], [8, 0, 0, .2], [0, 5, 0, .3]], np.float32).tofile(seq/'velodyne'/f'{index:06d}.bin')
                np.array([252, 10, 0], '<u4').tofile(seq/'labels'/f'{index:06d}.label')
            identity = '1 0 0 0 0 1 0 0 0 0 1 0'
            (seq/'poses.txt').write_text((identity+'\n')*2)
            (seq/'calib.txt').write_text('Tr: '+identity+'\n')
            checkpoint = root/'checkpoint.pt'
            torch.save(dict(cfg=config(), epoch=85, model_state_dict=self.model.state_dict()), checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args = ['--checkpoint', str(checkpoint), '--dataset-root', str(root/'dataset'), '--device', 'cpu',
                    '--mode', 'point', '--postprocess', 'knn', '--max-frames', '5']
            main(args+['--output-dir', str(root/'on')])
            main(args+['--output-dir', str(root/'off'), '--disable-semantic-gate'])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            rows = compare(root/'on', root/'off', root/'comparison')
            self.assertTrue(all(row['delta_on_minus_off'] == 0 for row in rows))
            metadata_path = root/'off/benchmark_eval_config.yaml'
            metadata = yaml.safe_load(metadata_path.read_text())
            self.assertEqual(metadata['checkpoint_model_config']['mos_model_params']['semantic_fusion'], 'bottleneck_gate')
            self.assertTrue(metadata['semantic_gate_ablation']['disable_semantic_gate'])
            self.assertEqual(metadata['n_scans'], 2)
            for key, bad in (('checkpoint', 'other.pt'), ('postprocessing', dict(method='knn', knn=3)),
                             ('checkpoint_epoch', 84), ('projection', {}), ('gt_mapping', 'wrong')):
                modified = deepcopy(metadata)
                modified[key] = bad
                metadata_path.write_text(yaml.safe_dump(modified))
                with self.assertRaisesRegex(ValueError, key):
                    compare(root/'on', root/'off', root/'bad')
            metadata_path.write_text(yaml.safe_dump(metadata))
            # Deliberately corrupt saved IoU: comparator must derive it from counts.
            path = root/'on/point_metrics.csv'
            with path.open() as handle:
                saved = list(csv.DictReader(handle))
            saved[0]['point_moving_iou'] = '0.123456'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(saved[0]))
                writer.writeheader(); writer.writerows(saved)
            repeated = compare(root/'on', root/'off', root/'counts_comparison')
            self.assertEqual(repeated[0]['gate_on'], rows[0]['gate_on'])
            with self.assertRaises(FileExistsError):
                compare(root/'on', root/'off', root/'comparison')

    def test_comparator_nonzero_deltas_from_counts(self):
        # Build minimal paired records to check the sign and percentage-point scale.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common = dict(checkpoint='/same.pt', checkpoint_sha256='same', checkpoint_epoch=85,
                          checkpoint_model_config=config(), eval_sequences=['08'], dataset_root='/data',
                          n_scans=1, moving_threshold=.5, projection={}, normalization='paper_normalization',
                          input_mode='range_xyz_intensity', history_init='zero_missing', pose_source='same',
                          gt_mapping='same', postprocessing={'method': 'knn', 'knn': 5}, partial_smoke_test=True,
                          pipeline_git_commit='same')
            for mode, counts in [('on', (8, 2, 2, 8)), ('off', (6, 4, 4, 6))]:
                folder = root/mode; folder.mkdir()
                meta = deepcopy(common)
                meta['semantic_gate_ablation'] = semantic_gate_ablation_metadata(self.model, mode == 'off')
                (folder/'benchmark_eval_config.yaml').write_text(yaml.safe_dump(meta))
                (folder/'projection_coverage.csv').write_text('sequence,frame,n_points\n08,000000,20\n')
                (folder/'point_metrics.csv').write_text('tp_moving,fp_moving,fn_moving,tn_static,n_gt_valid,n_points,n_scans\n'
                    + ','.join(map(str, (*counts, 20, 20, 1)))+'\n')
            rows = compare(root/'on', root/'off', root/'result')
            self.assertAlmostEqual(rows[0]['delta_on_minus_off'], 8/12-6/14)
            self.assertAlmostEqual(rows[0]['delta_percentage_points'], 100*(8/12-6/14))


if __name__ == '__main__':
    unittest.main()
