import os
from helper.egoMotion_compensation_calib import parse_calibration, parse_poses
from dataloader import RandomWindowSeqDataset
import torch
from torch.utils.data import DataLoader

def make_sequences(base_dir):
    """
    Walk each SequenceID folder in base_dir and return a list of dicts:
        {
            'seq_id':   sequence ID (folder name),
            'paths':    List of (pc_path, label_path) tuples,
            'poses':    List of 4x4 numpy arrays in Velodyne coords
        }
    """
    seqs = []
    for seq_id in sorted(os.listdir(base_dir)):
        seq_dir = os.path.join(base_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue

        vdir = os.path.join(seq_dir, 'velodyne')
        ldir = os.path.join(seq_dir, 'label')
        calfile = os.path.join(seq_dir, 'calib.txt')
        posefile = os.path.join(seq_dir, 'poses.txt')

        if not os.path.isdir(vdir):
            continue

        # parse calibration & poses
        calib = parse_calibration(calfile)
        poses = parse_poses(posefile, calib)

        # gather sorted file-pairs
        bins = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))
        paths = []
        for bf in bins:
            pc = os.path.join(vdir, bf)
            lb = os.path.join(ldir, bf.replace('.bin', '.label'))
            paths.append((pc, lb))

        # sanity check
        assert len(paths) == len(poses), \
            f"Mismatch in {seq_id}: {len(paths)} frames vs {len(poses)} poses"

        seqs.append({'seq_id': seq_id, 'paths': paths, 'poses': poses})
    return seqs

def build_dataloaders(seqs, cfg, device,
                      split_type='rotary', predefined_splits=None):
    """
    Create train/val (and optional test) DataLoaders from seq list.

    split_type:
      - 'rotary': leave-one-out on seq_ids each epoch (returns a list of tuples)
      - 'predefined': use explicit splits dict  

    predefined_splits: {
      'train': [seq_id, ...],
      'val':   [seq_id, ...],
      'test':  [seq_id, ...] (optional)
    }

    Returns:
      - rotary: list of (holdout_seq_id, train_loader, val_loader)
      - predefined: (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    idx_map = {s['seq_id']: i for i, s in enumerate(seqs)}

    if split_type == 'rotary':
        loaders = []
        for holdout in idx_map:
            val_idxs = [idx_map[holdout]]
            train_idxs = [i for i in range(len(seqs)) if i not in val_idxs]

            ds_train = RandomWindowSeqDataset([seqs[i] for i in train_idxs], cfg, device=device)
            ds_val = RandomWindowSeqDataset([seqs[i] for i in val_idxs], cfg, device=device)

            loader_train = DataLoader(
                ds_train,
                batch_size=cfg['train_params']['batch_size'],
                shuffle=True,
                num_workers=cfg['train_params']['dataloader_num_workers'],
                multiprocessing_context=torch.multiprocessing.get_context('spawn')
            )
            loader_val = DataLoader(
                ds_val,
                batch_size=cfg['train_params']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            loaders.append((holdout, loader_train, loader_val))
        return loaders

    elif split_type == 'predefined':
        assert predefined_splits is not None, "Need predefined_splits dict for 'predefined' mode"
        def ids(list_ids): return [idx_map[sid] for sid in list_ids]

        train_ids = ids(predefined_splits['train'])
        val_ids   = ids(predefined_splits['val'])
        test_ids  = ids(predefined_splits.get('test', []))

        ds_train = RandomWindowSeqDataset([seqs[i] for i in train_ids], cfg, device=device)
        ds_val   = RandomWindowSeqDataset([seqs[i] for i in val_ids], cfg, device=device)

        loader_train = DataLoader(
            ds_train,
            batch_size=cfg['train_params']['batch_size'],
            shuffle=True,
            num_workers=cfg['train_params']['dataloader_num_workers'],
            multiprocessing_context=torch.multiprocessing.get_context('spawn')
        )
        loader_val = DataLoader(
            ds_val,
            batch_size=cfg['train_params']['batch_size'],
            shuffle=False,
            num_workers=0
        )

        if test_ids:
            ds_test = RandomWindowSeqDataset([seqs[i] for i in test_ids], cfg, device=device)
            loader_test = DataLoader(
                ds_test,
                batch_size=cfg['train_params']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            return loader_train, loader_val, loader_test
        return loader_train, loader_val

    else:
        raise ValueError(f"Unknown split_type: {split_type}")