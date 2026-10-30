"""Parallel LMNet MAE: label independence, protocol transfer and accumulation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import train_temporal_e2e_mae as mae
import train_temporal_mos_e2e as mos
from helper.temporal_e2e_transfer import FORMAT, LMNET_FORMAT, input_metadata, validate_mae_checkpoint, load_mae_weights, HEAD
from temporal_e2e_mae_dataset import TemporalE2EMAEDataset, masked_input
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from test_lmnet_alignment import fixture
CONFIG=ROOT/'src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_lmnet_p1_2gpu.yaml'


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv2d(5,5,1)
    def forward(self,x,valid,dt): return self.conv(x[:,-1])


def toy_batches(rank):
    generator=torch.Generator().manual_seed(7+rank)
    batches=[]
    for i in range(3):
        x=torch.randn(1,2,5,16,32,generator=generator)
        valid=torch.ones(1,2,16,32,dtype=torch.bool)
        mask=torch.zeros(1,16,32,dtype=torch.bool); mask[:,:,:3+i*3+rank]=True
        batches.append(dict(x=x,valid=valid,dt=torch.tensor([[-.8,0.]]),mask=mask,target_current=x[:,-1].clone()))
    return batches


def ddp_accumulation(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'init'),rank=rank,world_size=2)
    try:
        cfg=tiny_config(directory); cfg['mae_train_params']['clip_grad_norm']=0
        torch.manual_seed(9)
        local=Toy(); reference=deepcopy(local)
        wrapped=mos.wrap_distributed(local,torch.device('cpu'))
        optimizer=torch.optim.SGD(wrapped.parameters(),lr=.001)
        ref_optimizer=torch.optim.SGD(reference.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        result=mae.phase(wrapped,toy_batches(rank),cfg,torch.device('cpu'),optimizer,scaler)
        assert result['samples']==6 and result['optimizer_steps']==2
        physical=deepcopy(cfg); physical['mae_train_params']['gradient_accumulation']=1
        all_batches=[toy_batches(r) for r in range(2)]
        for start in (0,2):
            group=[batch for batches in all_batches for batch in batches[start:start+2]]
            joined={key:torch.cat([v[key] for v in group]) for key in group[0]}
            mae.phase(reference,[joined],physical,torch.device('cpu'),ref_optimizer,scaler)
        for key,val in local.state_dict().items():
            torch.testing.assert_close(val,reference.state_dict()[key],rtol=1e-6,atol=1e-7)
    finally:
        dist.destroy_process_group()


def tiny_config(root):
    _,cfg=mae.read_config(CONFIG)
    cfg['model_params'].update(grid_height=16,grid_width=64)
    cfg['mae_params']['mask'].update(patch_h=8,patch_w=2)
    cfg['mos_data_params'].update(train_sequences=['00'],val_sequences=['08'])
    cfg['mae_train_params'].update(batch_size=1,gradient_accumulation=2,num_workers=0,epochs=3)
    cfg['mae_log_params'].update(log_root=str(Path(root)/'runs'),use_tensorboard=False)
    return cfg


class LMNetMAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_config_mask_proportions_and_p1_launcher(self):
        _,cfg=mae.read_config(CONFIG)
        old=yaml.safe_load((ROOT/'src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_p1.yaml').read_text())
        for axis,patch_key in [('grid_height','patch_h'),('grid_width','patch_w')]:
            self.assertEqual(cfg['mae_params']['mask'][patch_key]/cfg['model_params'][axis],
                             old['mae_params']['mask'][patch_key]/old['model_params'][axis])
        t=cfg['mae_train_params']
        self.assertEqual(t['batch_size']*2*t['gradient_accumulation'],16)
        self.assertEqual(t['epochs'],old['mae_train_params']['epochs'])
        self.assertFalse(t['amp'])
        with tempfile.TemporaryDirectory() as tmp:
            stub=Path(tmp)/'sbatch'
            stub.write_text('#!/bin/bash\nprintf "entry=%s logs=%s\\n" "$TEMPORAL_TRAIN_ENTRY" "$TEMPORAL_LOG_SUBDIR"\nprintf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'submit_temporal_e2e_mae_lmnet_p1_2gpu.sh')],
                env={**os.environ,'PATH':tmp+os.pathsep+os.environ['PATH'],'HPC_ROOT':tmp},capture_output=True,text=True,check=True)
            self.assertIn('--partition=p1',result.stdout)
            self.assertIn('--gres=gpu:2',result.stdout)
            self.assertIn('entry=tools/train_temporal_e2e_mae.py logs=temporal_e2e_mae_lmnet',result.stdout)
            self.assertIn(CONFIG.name,result.stdout)

    def test_label_free_input_matches_mos_and_boundary_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp); cfg=tiny_config(tmp)
            cfg['mos_augmentation']={'mode':'none'}
            cfg['mos_data_params']['previous_offsets']=[2,1]
            cfg['mos_data_params']['online_validation_labels']=True
            md=TemporalE2EMAEDataset([seq],cfg,'val')
            supervised=TemporalMOSE2EDataset([seq],cfg,'val')
            self.assertEqual(len(md),4)
            for i in range(4):
                a,b=md[i],supervised[i]
                for key,other in [('x',b[0]),('valid',b[1]),('dt',b[2])]:
                    torch.testing.assert_close(a[key],other,rtol=0,atol=0)
                torch.testing.assert_close(a['target_current'],a['x'][-1])
            self.assertFalse(md[0]['valid'][:-1].any())
            self.assertFalse(md[0]['x'][:-1].any())
            for path in (Path(tmp)/'00/labels').glob('*.label'): path.unlink()
            with patch('helper.mos_online.point_labels',side_effect=AssertionError('MAE must not load labels')):
                sample=md[2]
                self.assertTrue(torch.isfinite(sample['x']).all())
            self.assertTrue(torch.equal(md[2]['mask'],md[2]['mask']))
            x=sample['x'][None]; mask=sample['mask'][None]
            masked=masked_input(x,mask)
            self.assertTrue((masked[:,-1].masked_select(mask[:,None])==0).all())
            torch.testing.assert_close(masked[:,:-1],x[:,:-1])
            forbidden={**seq,'seq_id':'11'}
            with self.assertRaises(ValueError): TemporalE2EMAEDataset([forbidden],cfg,'train')

    def test_sync_augmentation_parity_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp); cfg=tiny_config(tmp)
            cfg['mos_data_params']['previous_offsets']=[2,1]
            md=TemporalE2EMAEDataset([seq],cfg,'train')
            sd=TemporalMOSE2EDataset([seq],cfg,'train')
            mos.seed_all(42); a=md[2]
            mos.seed_all(42); b=sd[2]
            torch.testing.assert_close(a['x'],b[0],rtol=0,atol=0)
            torch.testing.assert_close(a['valid'],b[1])
            torch.testing.assert_close(a['target_current'],a['x'][-1])

    def test_accumulated_objective_matches_physical_batch_without_batchnorm(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.conv=torch.nn.Conv2d(5,5,1)
            def forward(self,x,valid,dt): return self.conv(x[:,-1])
        with tempfile.TemporaryDirectory() as tmp:
            cfg=tiny_config(tmp)
            cfg['mae_train_params']['clip_grad_norm']=0
            generator=torch.Generator().manual_seed(7)
            batches=[]
            for i in range(3):
                x=torch.randn(1,2,5,16,32,generator=generator)
                valid=torch.ones(1,2,16,32,dtype=torch.bool)
                mask=torch.zeros(1,16,32,dtype=torch.bool); mask[:,:,:3+i*3]=True
                batches.append(dict(x=x,valid=valid,dt=torch.tensor([[-.8,0.]]),mask=mask,target_current=x[:,-1].clone()))
            for count in (1,2,3):
                a=Toy(); b=deepcopy(a)
                opta=torch.optim.SGD(a.parameters(),lr=.001); optb=torch.optim.SGD(b.parameters(),lr=.001)
                scaler=torch.amp.GradScaler('cuda',enabled=False)
                expected_steps=(count+1)//2
                accumulated=mae.phase(a,batches[:count],cfg,torch.device('cpu'),opta,scaler)
                for start in range(0,count,2):
                    group=batches[start:min(start+2,count)]
                    joined={key:torch.cat([v[key] for v in group]) for key in group[0]}
                    physical=deepcopy(cfg); physical['mae_train_params']['gradient_accumulation']=1
                    mae.phase(b,[joined],physical,torch.device('cpu'),optb,scaler)
                self.assertEqual(accumulated['optimizer_steps'],expected_steps)
                for key,val in a.state_dict().items(): torch.testing.assert_close(val,b.state_dict()[key],rtol=1e-6,atol=1e-7)

    def test_ddp_accumulation_matches_global_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_accumulation,args=(tmp,),nprocs=2,join=True)

    def test_versioned_transfer_resume_and_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp); cfg=tiny_config(tmp)
            cfg['mos_data_params']['previous_offsets']=[2,1]
            def setup():
                mos.seed_all(9)
                with patch.object(mos,'make_sequences',return_value=[seq,{**seq,'seq_id':'08'}]):
                    loaders=[mae.make_loader(cfg,s,torch.device('cpu')) for s in ('train','val')]
                model=mos.build_mos_model(cfg)
                opt=torch.optim.AdamW(model.parameters(),lr=.0005)
                return model,opt,mae.make_scheduler(opt,cfg['mae_train_params']),torch.amp.GradScaler('cuda',enabled=False),loaders
            model,opt,sched,scaler,loaders=setup()
            mae.set_epoch(loaders[0],0)
            mae.phase(model,loaders[0],cfg,torch.device('cpu'),opt,scaler,max_batches=2); sched.step()
            payload=mae.checkpoint_payload(model,opt,sched,scaler,cfg,1,.9,[mae.capture_rng(loaders)])
            self.assertEqual(payload['format'],LMNET_FORMAT)
            validate_mae_checkpoint(payload)
            path=Path(tmp)/'mae.pt'; torch.save(payload,path)
            mae.set_epoch(loaders[0],1)
            expected=mae.phase(model,loaders[0],cfg,torch.device('cpu'),opt,scaler,max_batches=2)
            clone,opt2,sched2,scaler2,loaders2=setup()
            mae.resume_training(path,clone,opt2,sched2,scaler2,cfg,loaders2,torch.device('cpu'))
            mae.set_epoch(loaders2[0],1)
            actual=mae.phase(clone,loaders2[0],cfg,torch.device('cpu'),opt2,scaler2,max_batches=2)
            self.assertEqual(actual,expected)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
            target_cfg=deepcopy(cfg); target_cfg['mos_model_params']['num_classes']=2
            target=mos.build_mos_model(target_cfg)
            head={k:v.clone() for k,v in target.state_dict().items() if k in HEAD}
            load_mae_weights(target,path,target_cfg)
            for k,v in torch.load(path,weights_only=False)['model_state_dict'].items():
                torch.testing.assert_close(target.state_dict()[k],head[k] if k in HEAD else v)
            for change in ('projection_mode','max_range','FOV_UP','previous_offsets','scan_rate_hz'):
                bad=deepcopy(target_cfg)
                if change=='projection_mode':
                    bad['mos_data_params'].pop('alignment_version'); bad['mos_data_params']['projection_mode']='legacy'; bad['mos_augmentation']={'mode':'none'}
                elif change=='previous_offsets': bad['mos_data_params'][change]=[1]
                elif change=='scan_rate_hz': bad['mos_data_params'][change]=20.
                else: bad['model_params'][change]=80. if change=='max_range' else 4.
                with self.subTest(change=change), self.assertRaises(ValueError): load_mae_weights(target,path,bad)
            _,legacy=mae.read_config(ROOT/'src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_p1.yaml')
            self.assertNotIn('input_protocol',input_metadata(legacy))
            old=mae.checkpoint_payload(model,opt,sched,scaler,legacy,1,.9,[])
            self.assertEqual(old['format'],FORMAT)
            validate_mae_checkpoint(old)

    def test_trainer_checkpoint_and_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp); cfg=tiny_config(tmp)
            cfg['mae_train_params']['epochs']=1
            path=Path(tmp)/'cfg.yaml'; path.write_text(yaml.safe_dump(cfg))
            args=SimpleNamespace(cfg_path=str(path),epochs=None,batch_size=None,num_workers=None,
                gradient_accumulation=None,log_root=None,resume=None,dry_run=False,max_batches=2,history_diagnostic=False)
            with patch.object(mos,'make_sequences',return_value=[seq,{**seq,'seq_id':'08'}]):
                run=mae.train(args,torch.device('cpu'))
            checkpoint=torch.load(run/'checkpoints/last.pt',weights_only=False)
            validate_mae_checkpoint(checkpoint)
            self.assertEqual(checkpoint['format'],LMNET_FORMAT)
            self.assertEqual(checkpoint['config']['run_metadata']['effective_global_batch_size'],2)
            self.assertEqual(checkpoint['config']['run_metadata']['gradient_accumulation'],2)
            self.assertIn('train/optimizer_steps',(run/'csv/epochs.csv').read_text())


if __name__=='__main__': unittest.main()
