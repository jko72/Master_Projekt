"""Residual auxiliary geometry, winner identity, regression, DDP and Slurm logs."""
from copy import deepcopy
import datetime
import os
from pathlib import Path
import subprocess
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
from helper.mos_auxiliary import auxiliary_settings, validate_auxiliary_config
from helper.mos_augmentation import TemporalMOSAugmentation, augmentation_settings
from helper.mos_projection import lmnet_projection
from helper.mos_residual_auxiliary import MaskedResidualSmoothL1, ResidualStatistics
from helper.residual_helper import compute_residual_target, transform_points
from helper.temporal_e2e_transfer import checkpoint_format, input_metadata, load_mae_weights
from mos_models import build_mos_model
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from test_lmnet_alignment import fixture
from test_temporal_mos_aux_movable import inputs

CPU = torch.device('cpu')
CONFIG = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml'
BASE = ROOT/'tests/reference/temporal_mos_lmnet_online_baseline.yaml'


def config():
    _, cfg = trainer.read_config(CONFIG)
    cfg['model_params'].update(grid_height=16, grid_width=64)
    return cfg


def projection(points):
    return lmnet_projection(np.asarray(points, np.float32).reshape(-1,4),16,64)


class ResidualDataset(torch.utils.data.Dataset):
    def __len__(self): return 10

    def __getitem__(self, i):
        x = torch.randn(2,5,2,3,generator=torch.Generator().manual_seed(i))
        y = torch.tensor([[i % 2,0,1],[-1,0,0]])
        target = torch.arange(6).reshape(2,3).float()*.1+i*.03
        mask = torch.arange(6).reshape(2,3) < i % 6
        if i in (0,1,4,9): mask.zero_()
        return x,torch.ones(2,2,3,dtype=torch.bool),torch.tensor([-.8,0.]),y,dict(
            task='residual',target=target,valid_mask=mask),dict(seq_id='00',frame_stem=str(i))


class ResidualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=torch.nn.Conv2d(5,4,1)
        self.mos=torch.nn.Conv2d(4,2,1)
        self.residual_head=torch.nn.Conv2d(4,1,1)

    def forward(self,x,valid,dt,return_aux=False):
        f=self.shared(x[:,-1]); mos=self.mos(f)
        return (mos,self.residual_head(f)) if return_aux else mos


def ddp_worker(rank,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{directory}/init',rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        cfg=config(); crit=trainer.make_criterion(cfg,CPU)
        model=trainer.wrap_distributed(ResidualModel(),CPU)
        reference=deepcopy(model.module)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        ref_opt=torch.optim.SGD(reference.parameters(),lr=.01)
        ds=ResidualDataset(); indices=([0,1,2,3,4],[5,6,7,8,9])
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices[rank]),batch_size=2)
        expected_count=0; expected_sum=0.; expected_absolute=0.; expected_squared=0.
        for group in ((0,1),(2,)):
            ref_opt.zero_grad(); group_residual=[]; group_valid=0
            for step in group:
                batches=[trainer.collate_mos([ds[i] for i in ids[2*step:2*step+2]]) for ids in indices]
                outputs=[reference(*b[:3],return_aux=True) for b in batches]
                logits=torch.cat([o[0] for o in outputs]); y=torch.cat([b[3] for b in batches])
                ce=crit.components(logits,y)['loss_ce']
                lovasz=torch.stack([crit.lovasz(o[0],b[3]) for o,b in zip(outputs,batches)]).mean()
                ((ce+lovasz)/len(group)).backward(retain_graph=True)
                for o,b in zip(outputs,batches):
                    aux=b[4]; parts=crit.residual_criterion.components(o[1],aux['target'],aux['valid_mask'])
                    group_residual.append(parts['loss_sum']); group_valid+=int(parts['valid_count'])
                    error=(o[1][:,0]-aux['target'])[aux['valid_mask']].detach().double()
                    expected_count+=int(parts['valid_count']); expected_sum+=float(parts['loss_sum'].detach())
                    expected_absolute+=float(error.abs().sum()); expected_squared+=float(error.square().sum())
            (.3*torch.stack(group_residual).sum()/max(group_valid,1)).backward()
            ref_opt.step()
        result=trainer.run_phase(model,loader,crit,CPU,optimizer,torch.amp.GradScaler('cuda',enabled=False),
                                 accumulation=2,clip=0)
        for actual,expected in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        assert result['aux_valid_pixels']==expected_count
        assert abs(result['aux_loss']-expected_sum/expected_count)<1e-6
        assert abs(result['aux_mae']-expected_absolute/expected_count)<1e-6
        assert abs(result['aux_rmse']-(expected_squared/expected_count)**.5)<1e-6
        assert abs(result['total_loss']-result['loss_total']-.3*result['aux_loss'])<1e-6
        val=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,[0] if rank==0 else []))
        result=trainer.run_phase(model,val,crit,CPU)
        assert result['samples']==1 and result['aux_loss']==0
        real=trainer.wrap_distributed(build_mos_model(cfg),CPU)
        for empty in (False,True):
            real.zero_grad(set_to_none=True)
            mos,residual=real(*inputs(),return_aux=True)
            y=torch.randint(0,2,(1,16,32)); target=torch.ones_like(y,dtype=torch.float32)
            mask=torch.full_like(y,not empty,dtype=torch.bool)
            (crit(mos,y)+.3*crit.residual_criterion(residual,target,mask)).backward()
            for p in real.parameters(): assert p.grad is not None and torch.isfinite(p.grad).all()
    finally: dist.destroy_process_group()


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_config_and_protocol(self):
        baseline=yaml.safe_load(BASE.read_text()); cfg=yaml.safe_load(CONFIG.read_text())
        cfg.pop('mos_auxiliary'); cfg['mos_log_params']['run_name']=baseline['mos_log_params']['run_name']
        self.assertEqual(cfg,baseline)
        for change in (dict(offset=0),dict(offset=True),dict(min_range=50),dict(beta=0),dict(epsilon=0),
                       dict(normalization='log1p'),dict(loss='ce'),dict(ignore_invalid_pairs=False),dict(typo=1)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                auxiliary_settings(dict(task='residual',residual=change))
        for offsets in ([7],[8,16],[]):
            cfg=config(); cfg['mos_data_params']['previous_offsets']=offsets
            with self.assertRaises(ValueError): validate_auxiliary_config(cfg)
        cfg=config(); cfg['mos_data_params'].pop('projection_mode'); cfg['mos_data_params'].pop('alignment_version')
        with self.assertRaises(ValueError): validate_auxiliary_config(cfg)
        with self.assertRaises(ValueError): auxiliary_settings(dict(task='residual',movable={}))

    def test_geometry_ranges_and_zero_validity(self):
        current=projection([[5,0,0,.2]])
        zero,valid=compute_residual_target(current,current)
        self.assertEqual(int(valid.sum()),1); self.assertEqual(float(zero[valid][0]),0.)
        past=projection([[20,0,0,.2]])
        target,valid=compute_residual_target(current,past)
        self.assertAlmostEqual(float(target[valid][0]),3.)  # deliberately >1
        _,valid=compute_residual_target(current,past,history_available=False); self.assertFalse(valid.any())
        for point in ([2,0,0,.2],[50,0,0,.2],[np.nan,0,0,.2],[np.inf,0,0,.2],[5,0,0,np.nan]):
            _,valid=compute_residual_target(projection([point]),past); self.assertFalse(valid.any())
        current_pose=np.eye(4); current_pose[0,3]=1
        past_pose=np.eye(4); past_scan=np.array([[10,0,0,.2]],np.float32)
        aligned=transform_points(past_scan,np.linalg.solve(current_pose,past_pose))
        target,valid=compute_residual_target(projection([[9,0,0,.2]]),projection(aligned))
        self.assertEqual(float(target[valid][0]),0)
        # Full-range projection precedes target filtering: no hidden far winner.
        _,valid=compute_residual_target(projection([[1,0,0,.2],[10,0,0,.2]]),past)
        self.assertFalse(valid.any())

    def test_dropout_winner_identity_geometry_and_rng(self):
        scans=[np.array([[4,0,0,.2]],np.float32),np.array([[5,0,0,.2],[10,0,0,.2]],np.float32)]
        aug=TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),'train')
        for drops,expected in (([0],False),([1],True),([],True)):
            params=dict(flip=True,translation=[0,0,0],drop_indices=drops)
            dirty,_,details=aug.raw_scans(scans,np.arange(2),params,return_details=True)
            clean=details['clean_scans']
            target,valid=compute_residual_target(projection(clean[-1]),projection(clean[0]),
                current_input=projection(dirty[-1]),input_original_ids=details['current_original_ids'])
            self.assertEqual(bool(valid.any()),expected)
            if expected:self.assertAlmostEqual(float(target[valid][0]),.2)
        params=dict(flip=True,translation=[1,2,-.5],drop_indices=[1])
        dirty,_,details=aug.raw_scans(scans,np.arange(2),params,return_details=True)
        for original,transformed in zip(scans,details['clean_scans']):
            expected=original.copy();expected[:,1]*=-1;expected[:,:3]+=[1,2,-.5]
            np.testing.assert_array_equal(transformed,expected)
        torch.manual_seed(42);old=aug.raw_scans(scans,np.arange(2));rng=torch.get_rng_state()
        torch.manual_seed(42);new=aug.raw_scans(scans,np.arange(2),return_details=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for a,b in zip(old[0],new[0]):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(old[1],new[1])

    def test_dataset_and_missing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,points=fixture(tmp);cfg=config();cfg['mos_data_params']['previous_offsets']=[1];cfg['mos_auxiliary']['residual']['offset']=1
            ds=TemporalMOSE2EDataset([seq],cfg,'val')
            first=ds[0];self.assertFalse(first[4]['valid_mask'].any())
            with patch('torch.rand',side_effect=AssertionError('validation drew RNG')):a=ds[2];b=ds[2]
            self.assertEqual(a[0].shape,(2,5,16,64));self.assertEqual(a[4]['target'].dtype,torch.float32)
            self.assertEqual(a[4]['valid_mask'].dtype,torch.bool)
            torch.testing.assert_close(a[4]['target'],b[4]['target'])
            pixel = projection(points).indices == 1  # current 5m, aligned history 4.9m
            self.assertTrue(a[4]['valid_mask'][pixel].all())
            self.assertAlmostEqual(float(a[4]['target'][pixel][0]), .1/5, places=6)
            self.assertLess(float(a[0][-1,3][pixel][0]), 0.)  # paper-normalized input != metres
            baseline=deepcopy(cfg);baseline.pop('mos_auxiliary')
            base=TemporalMOSE2EDataset([seq],baseline,'val')[2]
            for aa,bb in zip(a[:4],base[:4]):torch.testing.assert_close(aa,bb)
            # No extra random draws and no MOS-input changes under raw augmentation.
            aux=TemporalMOSE2EDataset([seq],cfg,'train');base=TemporalMOSE2EDataset([seq],baseline,'train')
            torch.manual_seed(42);base_item=base[2];state=torch.get_rng_state()
            torch.manual_seed(42);aux_item=aux[2];self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for aa,bb in zip(aux_item[:4],base_item[:4]):torch.testing.assert_close(aa,bb)
            test_seq,_=fixture(tmp,'11');test=TemporalMOSE2EDataset([test_seq],cfg,'test')
            with patch('helper.residual_helper.compute_residual_target',side_effect=AssertionError('inference built target')):
                item=test[2]
            self.assertEqual(item[3:5],(None,None));self.assertEqual(len(trainer.collate_mos([item])),6)

    def test_loss_empty_mask_fp32_and_invalid_nan(self):
        criterion=MaskedResidualSmoothL1(.1)
        p=torch.tensor([[[[0.,.2,99.]]]],requires_grad=True)
        target=torch.tensor([[[0.,.4,float('nan')]]]);mask=torch.tensor([[[True,True,False]]])
        with torch.autocast('cpu',dtype=torch.bfloat16):parts=criterion.components(p,target,mask)
        self.assertEqual(parts['loss_total'].dtype,torch.float32)
        expected=torch.nn.functional.smooth_l1_loss(p[0,0,0,:2],target[0,0,:2],beta=.1)
        torch.testing.assert_close(parts['loss_total'],expected)
        parts['loss_total'].backward();self.assertEqual(float(p.grad[0,0,0,2]),0)
        p.grad=None;loss=criterion(p,target,torch.zeros_like(mask));loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertTrue((p.grad==0).all())

    def test_model_gradient_and_strict_keys(self):
        cfg=config();base_cfg=deepcopy(cfg);base_cfg.pop('mos_auxiliary')
        torch.manual_seed(42);base=build_mos_model(base_cfg).eval();state=torch.get_rng_state()
        torch.manual_seed(42);model=build_mos_model(cfg).eval()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(set(model.state_dict())-set(base.state_dict()),{'residual_head.weight','residual_head.bias'})
        model.load_state_dict(model.state_dict(),strict=True);base.load_state_dict(base.state_dict(),strict=True)
        f=[];a=model.decoder.logits.register_forward_pre_hook(lambda m,args:f.append(args[0]))
        b=model.residual_head.register_forward_pre_hook(lambda m,args:f.append(args[0]))
        x=inputs();mos,aux=model(*x,return_aux=True)
        self.assertIs(f[0],f[1]);self.assertEqual(f[0].shape[1],32);a.remove();b.remove()
        self.assertEqual(aux.shape,(1,1,16,32));self.assertEqual(mos.shape,(1,2,16,32))
        torch.testing.assert_close(mos,base(*x),rtol=0,atol=0)
        with patch.object(model.residual_head,'forward',side_effect=AssertionError('MOS inference called residual')):
            torch.testing.assert_close(model(*x),mos,rtol=0,atol=0)
        y=torch.ones(1,16,32);mask=torch.ones_like(y,dtype=torch.bool)
        MaskedResidualSmoothL1()(aux,y,mask).backward()
        self.assertIsNone(model.decoder.logits.weight.grad)
        for module in (model.encoder,model.temporal,model.decoder.up_block4,model.residual_head):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()))
        movable=deepcopy(cfg);movable['mos_auxiliary']={'enabled':True,'task':'movable'}
        with self.assertRaises(RuntimeError):build_mos_model(movable).load_state_dict(model.state_dict(),strict=True)
        model.zero_grad(set_to_none=True);mos,aux=model(*x,return_aux=True)
        crit=trainer.make_criterion(cfg,CPU);loss=crit(mos,y.long())+.3*crit.residual_criterion(aux,y,mask)
        loss.backward()
        for p in model.parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())

    def test_mae_transfer(self):
        cfg=config();model=build_mos_model(cfg);source_cfg=deepcopy(cfg);source_cfg.pop('mos_auxiliary')
        source_cfg['mos_model_params']['num_classes']=5;source=build_mos_model(source_cfg)
        fresh={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('decoder.logits.','residual_head.'))}
        payload=dict(format=checkpoint_format(source_cfg),config=source_cfg,metadata=input_metadata(source_cfg),epoch=1,model_state_dict=source.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mae.pt';torch.save(payload,path);report=load_mae_weights(model,path,cfg)
            self.assertEqual(set(report['excluded_keys']),set(fresh))
            for k,v in model.state_dict().items():torch.testing.assert_close(v,fresh[k] if k in fresh else source.state_dict()[k])

    def test_checkpoint_train_reload_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_seq,_=fixture(tmp,'00');val_seq,_=fixture(tmp,'08');cfg=config()
            cfg['mos_data_params'].update(train_sequences=['00'],previous_offsets=[1]);cfg['mos_auxiliary']['residual']['offset']=1
            cfg['mos_train_params'].update(gradient_accumulation=2,batch_size=1)
            cfg['mos_log_params'].update(log_root=str(Path(tmp)/'runs'),use_tensorboard=False)
            path=Path(tmp)/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
            args=SimpleNamespace(cfg_path=str(path),log_root=None,pretrained_path=None,resume=None,dry_run=False,
                                 epochs=1,batch_size=1,num_workers=0,max_batches=3)
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]):trainer.train(args,CPU)
            run=next((Path(tmp)/'runs').iterdir());payload=torch.load(run/'checkpoints/last.pt',weights_only=False)
            for name in ('best_mean_iou.pt','best_moving_iou.pt'):self.assertTrue((run/'checkpoints'/name).is_file())
            self.assertEqual(payload['auxiliary_metadata']['task'],'residual')
            model=build_mos_model(payload['config']);model.load_state_dict(payload['model_state_dict'],strict=True)
            eval_cfg=deepcopy(cfg);eval_cfg['mos_auxiliary']={'enabled':False};eval_cfg['mos_augmentation']={'enabled':False}
            ds=TemporalMOSE2EDataset([val_seq],eval_cfg,'val');loader=torch.utils.data.DataLoader(ds,collate_fn=trainer.collate_mos)
            with patch.object(model.residual_head,'forward',side_effect=AssertionError('evaluator used auxiliary')):
                self.assertEqual(evaluate(model,loader,trainer.make_criterion(cfg,CPU),CPU,max_batches=1)['samples'],1)
            args.resume=str(run/'checkpoints/last.pt');args.epochs=2;args.log_root=str(Path(tmp)/'resumed')
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]):trainer.train(args,CPU)
            cfg['mos_auxiliary']['residual']['beta']=.2;path.write_text(yaml.safe_dump(cfg));args.log_root=str(Path(tmp)/'bad')
            with patch.object(trainer,'make_sequences',return_value=[train_seq,val_seq]),self.assertRaisesRegex(ValueError,'mos_auxiliary'):
                trainer.train(args,CPU)

    def test_statistics_histogram(self):
        cfg=config();stats=ResidualStatistics(cfg['mos_auxiliary']['residual'],CPU)
        target=torch.tensor([[[0.,.1,1.,3.]]]);mask=torch.ones_like(target,dtype=torch.bool);p=target[:,None]+.1
        parts=MaskedResidualSmoothL1().components(p,target,mask);stats.update(p,target,mask,parts)
        result=stats.finish(1.,.3)
        self.assertAlmostEqual(result['aux_mae'],.1,places=6);self.assertAlmostEqual(result['aux_rmse'],.1,places=6)
        self.assertEqual(result['aux_valid_pixels'],4)
        self.assertAlmostEqual(result['aux_target_mean'],1.025,places=6)
        for name,q in (('p50',.5),('p90',.9),('p99',.99)):
            expected=float(torch.quantile(target,q))
            self.assertLess(abs(result['aux_target_'+name]-expected),result['aux_target_quantile_bin_width'])

    def test_slurm_submit_log_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/'sbatch';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
            env={**os.environ,'PATH':tmp+os.pathsep+os.environ['PATH'],'HPC_ROOT':tmp,'P2_GPU_GRES':'gpu:2'}
            result=subprocess.run(['bash',str(ROOT/'submit_mos_p2_2gpu.sh'),str(CONFIG.relative_to(ROOT))],
                                  env=env,check=True,text=True,capture_output=True)
            args=result.stdout.splitlines()
            self.assertIn('--output='+tmp+'/logs/slurm/%x-%j.out',args)
            self.assertIn('--error='+tmp+'/logs/slurm/%x-%j.err',args)
            self.assertIn('--job-name='+CONFIG.stem,args)
            self.assertTrue((Path(tmp)/'logs/slurm').is_dir())
            self.assertIn('--gres=gpu:2',args)
            subprocess.run(['bash','-n',str(ROOT/'train_mos_p2_2gpu.slurm')],check=True)

    def test_distributed_accumulation_reference(self):
        with tempfile.TemporaryDirectory() as tmp:mp.spawn(ddp_worker,args=(tmp,),nprocs=2,join=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_amp_backward(self):
        cfg=config();model=build_mos_model(cfg).cuda();crit=trainer.make_criterion(cfg,torch.device('cuda'))
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for amp in (False,True):
            x=[v.cuda() for v in inputs()];y=torch.randint(0,2,(1,16,32),device='cuda')
            scaler=torch.amp.GradScaler('cuda',enabled=amp)
            trainer.dry_run_step(model,*x,y,crit,optimizer,scaler,torch.device('cuda'),amp,torch.float16)


if __name__=='__main__':unittest.main()
