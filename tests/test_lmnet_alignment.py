"""LMNet opt-in parity, geometry, cache isolation and optimizer regression tests."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'tests/reference')]
from helper.mos_projection import lmnet_projection, project_features, project_targets, projection_mode
from helper.mos_point_projection import ProjectionConfig, original_points_to_pixels
from helper.mos_label_helper import project_mos_labels_to_range, map_semantickitti_to_mos
from helper.mos_augmentation import TemporalMOSAugmentation, augmentation_settings
from helper.mos_input import normalization, normalize_feature_tensor, PAPER_MEAN, PAPER_STD
from helper.mos_cache import check_cache
from helper.residual_helper import project_range_for_residual, compute_residual_image
from helper.mos_optimization import LMNetWarmup, make_optimizer
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from mos_dataset import MOSFrameDataset
from temporal_mae_dataset import project_range_features
from evaluate_mos_benchmark import temporal_input, frame_features
from train_temporal_mos_e2e import read_config, collate_mos
from lmnet_warmupLR import warmupLR
from lmnet_laserscan import LaserScan
CONFIG = ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1.yaml'


def config():
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['model_params'].update(grid_height=16, grid_width=64)
    cfg['mos_data_params'].update(previous_offsets=[2, 1], online_validation_labels=True)
    return cfg


def fixture(root, seq_id='00'):
    directory = Path(root)/seq_id
    (directory/'velodyne').mkdir(parents=True)
    if int(seq_id) < 11:
        (directory/'labels').mkdir()
    points = np.array([[10, 0, 0, .21], [5, 0, 0, .3], [0, 100, 0, .6],
                       [0, -0.05, 0, .4], [-5, 0, 0, .2], [5, 0, 20, .1]], np.float32)
    labels = np.array([10, 252, 40, 10, 251, 0], np.uint32) | (np.uint32(7)<<16)
    paths = []
    poses = []
    for i in range(4):
        path = directory/'velodyne'/f'{i:06d}.bin'
        points.tofile(path)
        if int(seq_id) < 11:
            labels.tofile(directory/'labels'/f'{i:06d}.label')
        paths.append(str(path))
        pose = np.eye(4); pose[0, 3] = .1*i
        poses.append(pose)
    return dict(seq_id=seq_id, paths=paths, poses=poses), points


class ProjectionTests(unittest.TestCase):
    def test_author_parity_and_index_zero(self):
        rng = np.random.default_rng(9)
        points = rng.normal(size=(1000, 4)).astype(np.float32)*10
        points = np.concatenate([points, np.array([[1,0,0,.2], [2,0,0,.4],
            [0,1,0,.3], [0,-1,0,.3], [-1,0,0,.3], [1,0,100,.3]], np.float32)])
        ref = LaserScan(project=True, H=64, W=2048, fov_up=3, fov_down=-25)
        ref.set_points(points[:, :3].copy(), points[:, 3].copy())
        proj = lmnet_projection(points, 64, 2048)
        np.testing.assert_array_equal(proj.indices, ref.proj_idx)
        np.testing.assert_array_equal(proj.rows, ref.proj_y)
        np.testing.assert_array_equal(proj.cols, ref.proj_x)
        one = lmnet_projection(points[:1], 64, 2048)
        self.assertEqual(one.occupied.sum(), 1)
        self.assertIn(0, one.indices)

    def test_feature_target_correspondence_and_filters(self):
        points = np.array([[100,0,0,.2], [0,.05,0,.3], [0,0,0,.3],
                           [np.nan,0,0,.1], [0,-2,0,np.nan], [200,0,0,.1]], np.float32)
        feature, mask, proj = project_features(points, 64, 2048, np.deg2rad([-25,3]))
        labels = np.array([1,0,0,0,0,0])
        target = project_targets(labels, proj)
        np.testing.assert_array_equal(proj.usable, [True,True,False,False,False,True])
        self.assertEqual(float(feature[3].max()), 100.)
        np.testing.assert_array_equal(target[proj.occupied], labels[proj.indices[proj.occupied]])
        cfg = config(); cfg['model_params'].update(grid_height=64, grid_width=2048)
        np.testing.assert_array_equal(project_mos_labels_to_range(points, labels, cfg), target)
        rows, cols, usable = original_points_to_pixels(points, ProjectionConfig(64,2048,3,-25,projection_mode='lmnet'))
        np.testing.assert_array_equal(rows, proj.rows)
        np.testing.assert_array_equal(cols, proj.cols)
        np.testing.assert_array_equal(usable, proj.usable)
        self.assertTrue(torch.all(feature[:,~mask[0].bool()] == 0))

    def test_residual_filter_precedes_collision(self):
        cfg = config(); cfg['residual_params'] = dict(min_range=2., max_range=50., normalize=True)
        # Near invalid point must not hide valid 10m point at identical angle.
        pts = np.array([[1,0,0,.1], [10,0,0,.2], [50,0,0,.4], [2,0,0,.4]], np.float32)
        image = project_range_for_residual(pts, cfg)
        self.assertEqual(float(image.max()), 10.)
        past = np.array([[8,0,0,.1]], np.float32)
        diff = compute_residual_image(pts, past, np.eye(4), np.eye(4), cfg, 1)
        self.assertAlmostEqual(float(diff.max()), .2)

    def test_all_named_normalizations_and_knn_metres(self):
        cfg = config(); points = np.array([[10,2,-1,.3]], np.float32)
        f, m, _ = project_features(points,16,64,np.deg2rad([-25,3]))
        normalized = normalize_feature_tensor(f,m,normalization(cfg,'range_xyz_intensity'))
        for j, name in enumerate(['x','y','z','range','intensity']):
            torch.testing.assert_close(normalized[j,m[0].bool()], (f[j,m[0].bool()]-PAPER_MEAN[name])/PAPER_STD[name])
        channels, raw = frame_features(points,cfg,cfg['mos_data_params'],True)
        np.testing.assert_array_equal(raw,f[3].numpy())
        np.testing.assert_allclose(np.stack(channels),normalized.numpy())
        self.assertTrue(torch.all(normalized[:,~m[0].bool()] == 0))


class DatasetTests(unittest.TestCase):
    def test_dataset_draw_after_egomotion_and_custom_mode(self):
        from helper.residual_helper import transform_points
        with tempfile.TemporaryDirectory() as tmp:
            seq,points=fixture(tmp); cfg=config()
            ds=TemporalMOSE2EDataset([seq],cfg,'train')
            params=dict(flip=True,translation=[1,2,-.5],drop_indices=[1,1,3])
            raw=ds.augmentation.raw_scans
            with patch.object(ds.augmentation,'raw_scans',side_effect=lambda s,l:raw(s,l,params)):
                x,valid,dt,y,_=ds[2]
            aligned=[transform_points(points,np.linalg.solve(seq['poses'][2],seq['poses'][2-off])) for off in ds.offsets]+[points]
            labels=map_semantickitti_to_mos(np.array([10,252,40,10,251,0]))
            transformed,labels=raw(aligned,labels,params)
            for j,scan in enumerate(transformed):
                f,m,p=project_features(scan,ds.H,ds.W,ds.theta_range)
                torch.testing.assert_close(x[j],normalize_feature_tensor(f,m,normalization(cfg,'range_xyz_intensity')))
                torch.testing.assert_close(valid[j],m[0].bool())
                if j==2: np.testing.assert_array_equal(y.numpy(),project_targets(labels,p))
            cfg['mos_data_params']['normalization']='none'
            cfg['mos_augmentation']={'mode':'none'}
            base=TemporalMOSE2EDataset([seq],cfg,'train')[2]
            cfg['mos_augmentation']={'mode':'custom','azimuth_rotation':{'enabled':True,'probability':1.}}
            custom=TemporalMOSE2EDataset([seq],cfg,'train')
            with patch('torch.randint',return_value=torch.tensor(3)):
                changed=custom[2]
            expected=TemporalMOSAugmentation.rotate(list(base[0]),[m[None] for m in base[1]],base[3],3)
            torch.testing.assert_close(changed[0],torch.stack(expected[0]))
            torch.testing.assert_close(changed[3],expected[2])

    def test_raw_augmentation_injected_geometry_and_labels(self):
        aug = TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),'train')
        points = np.arange(24,dtype=np.float32).reshape(6,4)
        params = dict(flip=True,translation=[1,2,-.5],drop_indices=[1,1,3])
        scans, labels = aug.raw_scans([points,points],np.arange(6),params)
        expected = points.copy(); expected[:,1]*=-1; expected[:,:3]+=[1,2,-.5]
        np.testing.assert_array_equal(scans[0],expected)
        np.testing.assert_array_equal(scans[-1],expected[[0,2,4,5]])
        np.testing.assert_array_equal(labels,[0,2,4,5])
        for split in ('val','test'):
            val = TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),split)
            result, _ = val.raw_scans([points],np.arange(6),params)
            np.testing.assert_array_equal(result[0], points)
        for n in (0,1):
            with patch('torch.rand', return_value=torch.tensor(0.)):
                scans,_=aug.raw_scans([points[:n]],np.arange(n))
                self.assertEqual(len(scans[0]),n)
        with self.assertRaises(ValueError):
            augmentation_settings(dict(mode='lmnet', intensity_jitter=dict(enabled=True)))
        self.assertFalse(TemporalMOSAugmentation(augmentation_settings({'mode':'lmnet'}),'train').applies('azimuth_rotation'))

    def test_boundary_frames_and_train_eval_benchmark_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq, points = fixture(tmp)
            cfg=config(); cfg['mos_augmentation']={'mode':'none'}
            train=TemporalMOSE2EDataset([seq],cfg,'train')
            val=TemporalMOSE2EDataset([seq],cfg,'val')
            self.assertEqual(len(train),4)
            for i in range(4):
                a,b=train[i],val[i]
                for x,y in zip(a[:4],b[:4]): torch.testing.assert_close(x,y)
                inputs, _, raw=temporal_input([Path(p) for p in seq['paths']],i,seq['poses'],cfg,'strict',True,True)
                for x,y in zip(a[:3],inputs): torch.testing.assert_close(x,y[0])
                for j,off in enumerate(train.offsets):
                    if i<off:
                        self.assertFalse(a[1][j].any()); self.assertFalse(a[0][j].any())
                torch.testing.assert_close(a[2],torch.tensor([-.2,-.1,0.]))
            self.assertFalse((Path(tmp)/'00/mos_labels_2048_lmnet_v1').exists())
            # Baseline shares exact current input/label geometry with temporal.
            ds=MOSFrameDataset([seq],cfg,'val',input_mode='range_xyz_intensity')
            x,y,_=ds[0]
            torch.testing.assert_close(x,val[0][0][-1]); torch.testing.assert_close(y,val[0][3])

    def test_test_split_without_any_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp,'11'); cfg=config()
            ds=TemporalMOSE2EDataset([seq],cfg,'test')
            item=ds[0]
            self.assertIsNone(item[3]); self.assertEqual(len(ds),4)
            self.assertIsNone(collate_mos([item])[3])

    def test_cache_mismatch_and_cached_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq,_=fixture(tmp); cfg=config(); cfg['mos_augmentation']={'mode':'none'}
            ds=TemporalMOSE2EDataset([seq],cfg,'train')
            directory=Path(tmp)/'00'; folder=directory/cfg['mos_data_params']['mos_label_folder']
            with self.assertRaises(FileNotFoundError): check_cache(folder,cfg,'labels',directory)
            check_cache(folder,cfg,'labels',directory,create=True)
            for i in range(4): np.save(folder/f'{i:06d}.npy',ds[i][3].numpy())
            cfg['mos_data_params']['online_validation_labels']=False
            val=TemporalMOSE2EDataset([seq],cfg,'val')
            torch.testing.assert_close(ds[0][3],val[0][3])
            np.save(folder/'000000.npy',np.full((16,64),.5,np.float32))
            with self.assertRaises(ValueError): val[0]
            changed=copy.deepcopy(cfg); changed['model_params']['grid_width']=128
            with self.assertRaises(ValueError): check_cache(folder,changed,'labels',directory)
            changed=copy.deepcopy(cfg); changed['mos_data_params']['normalization']='intensity_only'
            check_cache(folder,changed,'labels',directory)  # normalization is not geometry
            (folder/'manifest.json').unlink()
            with self.assertRaises(ValueError): check_cache(folder,cfg,'labels',directory,create=True)

    def test_legacy_defaults_untouched(self):
        _,cfg=read_config(ROOT/'src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml')
        self.assertEqual(projection_mode(cfg),'legacy')
        self.assertEqual(cfg['mos_train_params']['optimizer'],'adamw')
        self.assertEqual(cfg['mos_train_params'].get('checkpoint_metric','moving_iou'),'moving_iou')
        self.assertEqual(cfg['mos_data_params']['normalization'],'intensity_only')
        points=np.array([[10,0,0,.2]],np.float32)
        a=project_range_features(points,16,64,np.deg2rad([-25,3]),.1,80)
        b=project_range_features(points,16,64,np.deg2rad([-25,3]),.1,80,projection_mode='legacy')
        for x,y in zip(a,b): torch.testing.assert_close(x,y)


class OptimizationTests(unittest.TestCase):
    def test_weights_from_pinned_authors(self):
        ref=yaml.safe_load((ROOT/'tests/reference/lmnet_mos_labels.yaml').read_text())
        cfg=yaml.safe_load(CONFIG.read_text())
        expected=[1/(sum(v for k,v in ref['content'].items() if ref['learning_map'][k]==i)+.001) for i in (1,2)]
        actual=[cfg['mos_train_params'][k] for k in ('static_class_weight','moving_class_weight')]
        np.testing.assert_allclose(actual,expected,rtol=1e-12)
        self.assertAlmostEqual(actual[1]/actual[0],290.34279423391575)

    def test_lr_trace_author_parity_and_resume(self):
        for updates in (1,4,10):
            p=torch.nn.Parameter(torch.ones(())); q=torch.nn.Parameter(torch.ones(()))
            opt=torch.optim.SGD([p],lr=.01,momentum=.9)
            refopt=torch.optim.SGD([q],lr=.01,momentum=.9)
            ours=LMNetWarmup(opt,updates)
            ref=warmupLR(refopt,.01,updates,.9,.99**(1/updates))
            for step in range(updates*3+2):
                self.assertAlmostEqual(opt.param_groups[0]['lr'],refopt.param_groups[0]['lr'],places=13)
                opt.step(); refopt.step(); ours.step(); ref.step()
                if step==updates:
                    savedopt=copy.deepcopy(opt.state_dict()); saved=copy.deepcopy(ours.state_dict())
                    newopt=torch.optim.SGD([p],lr=.01,momentum=.9)
                    resumed=LMNetWarmup(newopt,updates)
                    newopt.load_state_dict(savedopt); resumed.load_state_dict(saved)
                    for _ in range(4):
                        self.assertEqual(opt.param_groups[0]['lr'],newopt.param_groups[0]['lr'])
                        ours.step(); resumed.step(); ref.step()


class IntegrationTests(unittest.TestCase):
    def test_generation_training_and_exact_epoch_resume(self):
        import generate_mos_labels
        import generate_residuals
        import train_temporal_mos_e2e as trainer
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'dataset'/'sequences'
            for seq_id in ('00','08','11'):
                seq,_=fixture(data,seq_id)
                directory=data/seq_id
                (directory/'calib.txt').write_text('Tr: '+ ' '.join(map(str,np.eye(4)[:3].reshape(-1)))+'\n')
                (directory/'poses.txt').write_text(''.join(' '.join(map(str,p[:3].reshape(-1)))+'\n' for p in seq['poses']))
            cfg=config()
            cfg['dataset_path']=str(data)
            cfg['mos_data_params'].update(train_sequences=['00'],val_sequences=['08'],test_sequences=['11'])
            cfg['mos_train_params'].update(epochs=2, batch_size=1, num_workers=0)
            cfg['mos_log_params'].update(log_root=str(root/'runs'),use_tensorboard=False)
            cfg['residual_params']=dict(enabled=True,mode='precompute', offsets=[1],min_range=2.,max_range=50.,
                normalize=True,folder_template='residual_images_lmnet_v1_{offset}')
            path=root/'config.yaml'
            def save(name):
                cfg['mos_log_params']['run_name']=name
                path.write_text(yaml.safe_dump(cfg))
            save('generation')
            with patch.object(sys,'argv',['labels','--cfg_path',str(path),'--seq_id','08','--max_frames','4']):
                generate_mos_labels.main()
            self.assertTrue((data/'08/mos_labels_2048_lmnet_v1/manifest.json').exists())
            with patch.object(sys,'argv',['labels','--cfg_path',str(path),'--seq_id','11','--max_frames','1']):
                generate_mos_labels.main()
            self.assertFalse((data/'11/mos_labels_2048_lmnet_v1').exists())
            with patch.object(sys,'argv',['residuals','--cfg_path',str(path),'--seq_id','00','--max_frames','3']):
                generate_residuals.main()
            self.assertFalse(np.load(data/'00/residual_images_lmnet_v1_1/000000.npy').any())
            from helper.dataloader_helper import make_sequences
            seqs=make_sequences(str(data))
            baseline_cfg=copy.deepcopy(cfg)
            baseline_cfg['mos_data_params'].update(input_mode='range_xyz_intensity_residual',residual_offsets=[1])
            baseline_cfg['mos_data_params']['residual_folder_template']='residual_images_lmnet_v1_{offset}'
            baseline_cfg['mos_augmentation']={'mode':'none'}
            baseline=MOSFrameDataset([s for s in seqs if s['seq_id']=='00'], baseline_cfg,
                                     'val',input_mode='range_xyz_intensity_residual',residual_offsets=[1])
            self.assertEqual(baseline[0][0].shape[0],6)
            from evaluate_mos_benchmark import single_frame_input
            projected=single_frame_input(data/'00/velodyne/000001.bin',baseline_cfg,data/'00','000001')
            torch.testing.assert_close(projected[0],baseline[1][0])
            with self.assertRaises(FileNotFoundError): baseline[3]
            cfg['mos_data_params']['online_validation_labels']=False
            args=SimpleNamespace(cfg_path=str(path),log_root=None,pretrained_path=None,
                epochs=None,batch_size=None,num_workers=None,max_batches=1,dry_run=False,
                diagnose_validation=False,resume=None)
            save('continuous')
            trainer.train(args,torch.device('cpu'))
            continuous=next((root/'runs').glob('continuous_*/checkpoints/last.pt'))
            save('first'); args.epochs=1
            trainer.train(args,torch.device('cpu'))
            first=next((root/'runs').glob('first_*/checkpoints/last.pt'))
            save('resumed'); args.epochs=2; args.resume=str(first)
            trainer.train(args,torch.device('cpu'))
            resumed=next((root/'runs').glob('resumed_*/checkpoints/last.pt'))
            a=torch.load(continuous,weights_only=False); b=torch.load(resumed,weights_only=False)
            for key,value in a['model_state_dict'].items():
                torch.testing.assert_close(value,b['model_state_dict'][key],rtol=0,atol=0)
            self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
            self.assertEqual(b['checkpoint_metric'],'mean_iou')
            self.assertTrue(resumed.with_name('best_mean_iou.pt').exists() or first.with_name('best_mean_iou.pt').exists())
            self.assertEqual(a['optimizer_state_dict']['param_groups'],b['optimizer_state_dict']['param_groups'])
            self.assertTrue(list(resumed.parents[1].glob('samples_rank*.jsonl')))
            import evaluate_mos_benchmark as benchmark
            for post in ('direct', 'knn'):
                benchmark.main(['--checkpoint',str(resumed),'--dataset-root',str(data),
                    '--sequences','08','--mode','both','--postprocess',post,'--device','cpu',
                    '--output-dir',str(root/post),'--max-frames','2'])
                metadata=yaml.safe_load((root/post/'benchmark_eval_config.yaml').read_text())
                self.assertEqual(metadata['history_init'],'zero_missing')
                self.assertEqual(metadata['n_scans'],2)
            benchmark.main(['--checkpoint',str(resumed),'--dataset-root',str(data),
                '--sequences','11','--mode','point','--postprocess','knn','--device','cpu',
                '--output-dir',str(root/'export'),'--export-predictions'])
            exported=list((root/'export/predictions').rglob('*.label'))
            self.assertEqual(len(exported),4)
            for path in exported:
                labels=np.fromfile(path,np.uint32)
                self.assertEqual(len(labels),6)
                self.assertTrue(np.isin(labels,[9,251]).all())


if __name__=='__main__': unittest.main()
