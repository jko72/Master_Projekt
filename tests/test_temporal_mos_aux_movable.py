"""Movable auxiliary supervision: raw targets, model, loss, transfer and DDP."""
from copy import deepcopy
import csv
import datetime
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
import train_temporal_mos_e2e as trainer
from evaluate_temporal_mos_e2e import evaluate
from helper.mos_auxiliary import auxiliary_settings, MOVABLE_CONTENT, MOVABLE_WEIGHTS
from helper.mos_augmentation import TemporalMOSAugmentation, augmentation_settings
from helper.mos_label_helper import map_semantickitti_to_movable, map_semantickitti_to_mos
from helper.mos_lovasz_loss import MOSCrossEntropyLovasz
from helper.mos_projection import project_features, project_targets
from helper.temporal_e2e_transfer import load_mae_weights, checkpoint_format, input_metadata
from mos_models import build_mos_model
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from test_lmnet_alignment import fixture

CONFIG = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml'
BASELINE = ROOT/'tests/reference/temporal_mos_lmnet_online_baseline.yaml'
CPU = torch.device('cpu')


def config():
    _, cfg = trainer.read_config(CONFIG)
    cfg['model_params'].update(grid_height=16, grid_width=64)
    return cfg


def inputs():
    return (torch.randn(1, 2, 5, 16, 32), torch.ones(1, 2, 16, 32, dtype=torch.bool),
            torch.tensor([[-.8, 0.]]))


class AuxDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, i):
        x = torch.randn(2, 5, 2, 3, generator=torch.Generator().manual_seed(i))
        mos = torch.tensor([[i % 2, 0, 1], [-1, 0, 0]])
        movable = torch.tensor([[1, -1, i % 2], [0, -1, 1]])
        # Different validity and class proportions; globally empty middle step.
        if i in (0, 1, 2, 3, 4, 5):
            movable.fill_(-1)
        return x, torch.ones(2, 2, 3, dtype=torch.bool), torch.tensor([-.8, 0.]), mos, movable, {'seq_id': '00', 'frame_stem': str(i)}


class AuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Conv2d(5, 4, 1)
        self.mos = torch.nn.Conv2d(4, 2, 1)
        self.movable_head = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x, valid, dt, return_aux=False):
        f = self.shared(x[:, -1])
        mos = self.mos(f)
        return (mos, self.movable_head(f)) if return_aux else mos


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/init', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        cfg = config()
        crit = trainer.make_criterion(cfg, CPU)
        model = trainer.wrap_distributed(AuxModel(), CPU)
        reference = deepcopy(model.module)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        ds = AuxDataset()
        indices = ([0, 1, 2, 3, 6], [8, 9, 4, 5, 7])
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, indices[rank]), batch_size=2)
        for group in ((0, 1), (2,)):
            ref_optimizer.zero_grad()
            for step in group:
                batches = [trainer.collate_mos([ds[i] for i in ids[2*step:2*step+2]]) for ids in indices]
                outputs = [reference(*b[:3], return_aux=True) for b in batches]
                loss = 0.
                for task, criterion in enumerate((crit, crit.movable_criterion)):
                    logits = torch.cat([o[task] for o in outputs])
                    targets = torch.cat([b[3+task] for b in batches])
                    ce = criterion.components(logits, targets)['loss_ce']
                    lovasz = torch.stack([criterion.lovasz(o[task], b[3+task]) for o,b in zip(outputs,batches)]).mean()
                    loss = loss + ce + lovasz
                (loss / len(group)).backward()
            ref_optimizer.step()
        result = trainer.run_phase(model, loader, crit, CPU, optimizer,
                                   torch.amp.GradScaler('cuda', enabled=False), accumulation=2, clip=0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert abs(result['multitask_loss_total'] - result['loss_total'] - result['movable_loss_total']) < 1e-6
        # Uneven validation including an entirely empty rank must finish.
        val = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [0] if rank == 0 else []))
        result = trainer.run_phase(model, val, crit, CPU)
        assert result['samples'] == 1 and result['movable_loss_total'] == 0
        # Actual temporal model, two backwards, all active parameters in reducer.
        real = trainer.wrap_distributed(build_mos_model(cfg), CPU)
        for _ in range(2):
            real.zero_grad(set_to_none=True)
            mos, movable = real(*inputs(), return_aux=True)
            y = torch.randint(0, 2, (1, 16, 32))
            (crit(mos, y) + crit.movable_criterion(movable, y)).backward()
            for p in real.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
    finally:
        dist.destroy_process_group()


class MovableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mapping(self):
        sem = np.array([10, 252, 30, 50, 16, 256, 0, 1, 12345], np.uint32)
        raw = sem | (np.uint32(73) << 16)
        np.testing.assert_array_equal(map_semantickitti_to_movable(raw), [1,1,1,0,0,1,-1,-1,-1])
        np.testing.assert_array_equal(map_semantickitti_to_mos(sem), [0,1,0,0,0,1,-1,-1,-1])
        from helper.mos_label_helper import MOVABLE_LABELS, NON_MOVABLE_LABELS
        self.assertTrue((map_semantickitti_to_movable(MOVABLE_LABELS) == 1).all())
        self.assertTrue((map_semantickitti_to_movable(NON_MOVABLE_LABELS) == 0).all())

    def test_configuration_and_unchanged_protocol(self):
        base = yaml.safe_load(BASELINE.read_text())
        new = yaml.safe_load(CONFIG.read_text())
        new.pop('mos_auxiliary')
        new['mos_log_params']['run_name'] = base['mos_log_params']['run_name']
        self.assertEqual(new, base)
        self.assertFalse(auxiliary_settings()['enabled'])
        bad = [dict(task='unknown'), dict(loss_weight=-1), dict(loss_weight=float('nan')),
               dict(enabled=1), dict(unknown=3), dict(inference={'use_auxiliary_output':True}),
               dict(inference={'typo':False}), dict(movable={'ignore_index':0}),
               dict(movable={'class_weights':[1,0]}), dict(movable={'class_weights':[1,2,3]}),
               dict(movable={'class_weights':[float('inf'),2]}), dict(movable={'typo':1}),
               dict(movable={'loss':{'type':'ce'}}), dict(movable={'loss':{'typo':1}})]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                auxiliary_settings(value)

    def test_joint_projection_collision_and_raw_augmentation(self):
        points = np.array([[10,0,0,.1],[5,0,0,.2],[0,5,0,.3],[0,0,0,.4]], np.float32)
        sem = np.array([252,10,50,0], np.uint32)
        labels = np.stack([map_semantickitti_to_mos(sem),map_semantickitti_to_movable(sem)],1)
        aug = TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),'train')
        params = dict(flip=True, translation=[0,0,0], drop_indices=[2,2])
        scans, target = aug.raw_scans([points, points], labels, params)
        np.testing.assert_array_equal(target, labels[[0,1,3]])
        f,m,p = project_features(scans[-1],16,64,np.deg2rad([-25,3]))
        images = np.stack([project_targets(target[:,i],p) for i in range(2)])
        np.testing.assert_array_equal(images[:,p.occupied], target[p.indices[p.occupied]].T)
        np.testing.assert_array_equal(images[:,p.occupied], [[0],[1]])  # near parked car wins both
        self.assertTrue((images[:,~p.occupied] == -1).all())
        _, one = aug.raw_scans([points], labels[:,0], params)
        np.testing.assert_array_equal(one, labels[[0,1,3],0])
        for invalid in (np.zeros((4,2,1)), np.zeros(3), np.zeros((4,0))):
            with self.assertRaises(ValueError): aug.raw_scans([points], invalid, params)
        with patch('torch.rand', side_effect=AssertionError('validation drew RNG')):
            v = TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),'val')
            same, same_labels = v.raw_scans([points], labels)
            np.testing.assert_array_equal(same[0],points)
            np.testing.assert_array_equal(same_labels,labels)

    def test_dataset_joint_rng_and_test_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq, _ = fixture(tmp)
            cfg = config()
            base_cfg = deepcopy(cfg); base_cfg.pop('mos_auxiliary')
            base = TemporalMOSE2EDataset([seq], base_cfg)
            aux = TemporalMOSE2EDataset([seq], cfg)
            torch.manual_seed(42); original = base[0]; state = torch.get_rng_state()
            torch.manual_seed(42); added = aux[0]
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for a,b in zip(original[:4],added[:4]): torch.testing.assert_close(a,b)
            self.assertEqual(len(original),5); self.assertEqual(len(added),6)
            self.assertTrue(set(added[4].unique().tolist()) <= {-1,0,1})
            self.assertTrue((added[4][~added[1][-1]] == -1).all())
            # Exactly one raw augmentation call and one labels read for both tasks.
            with patch.object(aux.augmentation,'raw_scans', wraps=aux.augmentation.raw_scans) as draw, \
                 patch('helper.mos_online.np.fromfile', wraps=np.fromfile) as reads:
                aux[0]
                self.assertEqual(draw.call_count,1)
                self.assertEqual(sum(str(c.args[0]).endswith('.label') for c in reads.call_args_list),1)
            val = TemporalMOSE2EDataset([seq],cfg,'val')
            with patch('torch.rand',side_effect=AssertionError('validation drew RNG')):
                a = val[0]; b = val[0]
            for aa,bb in zip(a[:5],b[:5]): torch.testing.assert_close(aa,bb)
            # Custom range-view augmentation must roll both targets in one draw.
            cfg['mos_augmentation'] = {'mode':'custom','azimuth_rotation':{'enabled':True,'probability':1.}}
            custom = TemporalMOSE2EDataset([seq],cfg)
            with patch('torch.randint', return_value=torch.tensor(3)):
                c = custom[0]
            for i in (3,4): torch.testing.assert_close(c[i],torch.roll(a[i],3,-1))
            test_seq, _ = fixture(tmp,'11')
            test = TemporalMOSE2EDataset([test_seq],cfg,'test')
            item = test[0]
            self.assertEqual(item[3:5],(None,None))
            batch = trainer.collate_mos([item,item]); self.assertEqual(len(batch),6)
            self.assertIsNone(batch[3]); self.assertIsNone(batch[4])

    def test_model_features_rng_keys_and_mos_inference(self):
        cfg = config(); base_cfg = deepcopy(cfg); base_cfg.pop('mos_auxiliary')
        torch.manual_seed(42); base = build_mos_model(base_cfg).eval(); rng = torch.get_rng_state()
        torch.manual_seed(42); aux = build_mos_model(cfg).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(aux.state_dict()) - set(base.state_dict()), {'movable_head.weight','movable_head.bias'})
        for k,v in base.state_dict().items(): torch.testing.assert_close(v,aux.state_dict()[k],rtol=0,atol=0)
        build_mos_model(base_cfg).load_state_dict(base.state_dict(),strict=True)
        x = inputs(); features = []
        h1=aux.decoder.logits.register_forward_pre_hook(lambda m,args:features.append(args[0]))
        h2=aux.movable_head.register_forward_pre_hook(lambda m,args:features.append(args[0]))
        with torch.no_grad():
            mos,movable=aux(*x,return_aux=True)
            self.assertEqual(tuple(mos.shape),(1,2,16,32)); self.assertEqual(movable.shape,mos.shape)
            self.assertIs(features[0],features[1]); self.assertEqual(features[0].shape[1],32)
            h1.remove(); h2.remove()
            expected=base(*x); torch.testing.assert_close(mos,expected,rtol=0,atol=0)
            with patch.object(aux.movable_head,'forward',side_effect=AssertionError('inference used auxiliary')):
                torch.testing.assert_close(aux(*x),expected,rtol=0,atol=0)
                out, diagnostic = aux(*x,return_diagnostics=True)
                torch.testing.assert_close(out,expected)
            aux.movable_head.weight.fill_(1000)
            torch.testing.assert_close(aux(*x),expected,rtol=0,atol=0)
            (out,mov),diagnostic=aux(*x,return_aux=True,return_diagnostics=True)
            self.assertIsInstance(diagnostic,dict)
        with self.assertRaises(ValueError): base(*x,return_aux=True)
        self.assertEqual(sum(p.numel() for p in aux.parameters())-sum(p.numel() for p in base.parameters()),66)

    def test_gradient_paths_and_losses(self):
        cfg=config(); model=build_mos_model(cfg).eval(); crit=trainer.make_criterion(cfg,CPU)
        torch.testing.assert_close(crit.movable_criterion.weight,torch.tensor(MOVABLE_WEIGHTS))
        np.testing.assert_allclose(1/(np.array(MOVABLE_CONTENT)+.001),MOVABLE_WEIGHTS,rtol=1e-10)
        authors = yaml.safe_load((ROOT/'tests/reference/mfmos_movable_statistics.yaml').read_text())
        content = [sum(f for k,f in authors['content'].items() if authors['movable_learning_map'][k] == c) for c in (1,2)]
        np.testing.assert_allclose(content,MOVABLE_CONTENT,rtol=1e-14)
        for semantic, cls in authors['movable_learning_map'].items():
            self.assertEqual(int(map_semantickitti_to_movable(semantic)),cls-1)
        mos,movable=model(*inputs(),return_aux=True)
        y=torch.randint(0,2,(1,16,32)); y[:,0]=-1
        parts=crit.movable_criterion.components(movable,y)
        torch.testing.assert_close(parts['loss_ce'],torch.nn.functional.cross_entropy(movable,y,weight=crit.movable_criterion.weight,ignore_index=-1))
        torch.testing.assert_close(parts['loss_total'],parts['loss_ce']+parts['loss_lovasz'])
        parts['loss_total'].backward()
        self.assertIsNone(model.decoder.logits.weight.grad)
        for module in (model.encoder,model.temporal,model.decoder.up_block4,model.movable_head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        model.zero_grad(set_to_none=True)
        mos,movable=model(*inputs(),return_aux=True)
        total=crit(mos,y)+crit.movable_criterion(movable,y)
        total.backward()
        for p in model.parameters(): self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
        empty=torch.full_like(y,-1)
        logits=torch.randn(1,2,16,32,requires_grad=True)
        loss=crit.movable_criterion(logits,empty); self.assertEqual(loss.item(),0)
        loss.backward(); self.assertTrue((logits.grad==0).all())

    def test_mae_strict_transfer_both_heads_fresh(self):
        cfg=config(); model=build_mos_model(cfg)
        source_cfg=deepcopy(cfg); source_cfg.pop('mos_auxiliary'); source_cfg['mos_model_params']['num_classes']=5
        source=build_mos_model(source_cfg)
        original={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('movable_head.','decoder.logits.'))}
        payload=dict(format=checkpoint_format(source_cfg),config=source_cfg,metadata=input_metadata(source_cfg),
                     model_state_dict=source.state_dict(),epoch=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mae.pt'; torch.save(payload,path)
            report=load_mae_weights(model,path,cfg)
            self.assertEqual(set(report['excluded_keys']),set(original))
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,original[k] if k in original else source.state_dict()[k])
            payload['model_state_dict']['unexpected.weight']=torch.ones(1); torch.save(payload,path)
            with self.assertRaises(ValueError): load_mae_weights(model,path,cfg)

    def test_train_validation_checkpoint_reload_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_seq,_=fixture(tmp,'00'); val_seq,_=fixture(tmp,'08')
            cfg=config(); cfg['mos_data_params']['train_sequences']=['00']; cfg['mos_log_params'].update(log_root=str(Path(tmp)/'runs'),use_tensorboard=False)
            cfg['mos_train_params'].update(batch_size=1,num_workers=0)
            path=Path(tmp)/'cfg.yaml'; path.write_text(yaml.safe_dump(cfg))
            args=SimpleNamespace(cfg_path=str(path),log_root=None,pretrained_path=None,resume=None,dry_run=False,
                                 epochs=1,batch_size=1,num_workers=0,max_batches=1)
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]):
                trainer.train(args,CPU)
            run=next((Path(tmp)/'runs').iterdir()); ckpt=run/'checkpoints/last.pt'
            payload=torch.load(ckpt,weights_only=False)
            self.assertEqual(payload['auxiliary_metadata'],auxiliary_settings(cfg['mos_auxiliary']))
            model=build_mos_model(payload['config']); model.load_state_dict(payload['model_state_dict'],strict=True)
            for name in ('best_mean_iou.pt','best_moving_iou.pt'): self.assertTrue((run/'checkpoints'/name).is_file())
            with (run/'csv/epochs.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            for phase in ('train','val'):
                r=rows[0]
                self.assertAlmostEqual(float(r[phase+'_multitask_loss_total']),float(r[phase+'_loss_total'])+float(r[phase+'_movable_loss_total']))
            eval_cfg=deepcopy(cfg); eval_cfg['mos_auxiliary']={'enabled':False}; eval_cfg['mos_augmentation']={'enabled':False}
            ds=TemporalMOSE2EDataset([val_seq],eval_cfg,'val')
            loader=torch.utils.data.DataLoader(ds,collate_fn=trainer.collate_mos)
            with patch.object(model.movable_head,'forward',side_effect=AssertionError('official evaluator used auxiliary')):
                result=evaluate(model,loader,trainer.make_criterion(cfg,CPU),CPU,max_batches=1)
            self.assertEqual(result['samples'],1)
            args.resume=str(ckpt); args.epochs=2; args.log_root=str(Path(tmp)/'resumed')
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]): trainer.train(args,CPU)
            resumed=next((Path(tmp)/'resumed').iterdir())
            self.assertEqual(torch.load(resumed/'checkpoints/last.pt',weights_only=False)['epoch'],2)
            cfg['mos_auxiliary']['loss_weight']=.5; path.write_text(yaml.safe_dump(cfg))
            args.log_root=str(Path(tmp)/'invalid-resume')
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]), self.assertRaisesRegex(ValueError,'mos_auxiliary'):
                trainer.train(args,CPU)

    def test_two_task_ddp_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker,args=(tmp,),nprocs=2,join=True)


if __name__=='__main__': unittest.main()
