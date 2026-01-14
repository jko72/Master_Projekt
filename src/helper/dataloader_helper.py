import os, random
from helper.egoMotion_compensation_calib import parse_calibration, parse_poses
from dataloader import RandomWindowSeqDataset
import torch
from torch.utils.data import DataLoader
import numpy as np

# ---- Worker seeding (top-level, picklable) ----
WORKER_BASE_SEED = None

def set_worker_seed_base(s: int | None):
    global WORKER_BASE_SEED
    WORKER_BASE_SEED = None if s is None else int(s)

def worker_init_fn(worker_id: int):
    import random, numpy as np, torch
    base = 0 if WORKER_BASE_SEED is None else WORKER_BASE_SEED
    seed = base + worker_id
    random.seed(seed)
    # modulo, damit der seed im gültigen Bereich für numpy liegt
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


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
        try: 
            _ = float(seq_id)
        except:
            # No valid dataset sequence split need to be numbers #TODO: to be changed if datasets require other directory names than numbers
            continue
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
        if os.path.isfile(calfile):
            calib = parse_calibration(calfile)
        else:
            calib = None    # assumed that calib['Tr'] is eye matrix
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
                      split_type='rotary', predefined_splits=None,
                      dataloader_device='cpu'):
    """
    device: model/training device (e.g. 'cuda')
    dataloader_device: dataset device (SHOULD be 'cpu' for speed)
    """
    idx_map = {s['seq_id']: i for i, s in enumerate(seqs)}

    seed = cfg.get('train_params', {}).get('random_seed', None)
    deterministic = cfg.get('train_params', {}).get('deterministic', False)
    shuffle_train = cfg.get('train_params', {}).get('shuffle_train', True)

    set_worker_seed_base(seed)

    g = None
    wi_fn = None
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)
        wi_fn = worker_init_fn   # <-- FIX: use the top-level function
    else:
        set_worker_seed_base(None)
        wi_fn = None

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    num_workers = int(cfg['train_params']['dataloader_num_workers'])
    pin_memory = (str(device).startswith("cuda"))  # pin memory helps only when training on GPU

    def _make_loader(ds, shuffle: bool, num_workers: int):
        return DataLoader(
            ds,
            batch_size=cfg['train_params']['batch_size'],
            shuffle=shuffle,
            num_workers=num_workers,
            # NOTE: spawn is more overhead; keep for now since you set it explicitly.
            multiprocessing_context=torch.multiprocessing.get_context('spawn'),
            generator=g,
            worker_init_fn=wi_fn,
            persistent_workers=(num_workers > 0),
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )

    if split_type == 'rotary':
        loaders = []
        for holdout in idx_map:
            val_idxs = [idx_map[holdout]]
            train_idxs = [i for i in range(len(seqs)) if i not in val_idxs]

            # IMPORTANT: dataset on CPU
            ds_train = RandomWindowSeqDataset([seqs[i] for i in train_idxs], cfg, device=dataloader_device)
            ds_val   = RandomWindowSeqDataset([seqs[i] for i in val_idxs],   cfg, device=dataloader_device)

            loader_train = _make_loader(ds_train, shuffle=shuffle_train, num_workers=num_workers)
            loader_val = DataLoader(
                ds_val,
                batch_size=cfg['train_params']['batch_size'],
                shuffle=False,
                num_workers=cfg['train_params']['batch_size'],
                pin_memory=pin_memory
            )
            loaders.append((holdout, loader_train, loader_val))
        return loaders

    elif split_type == 'predefined':
        assert predefined_splits is not None, "Need predefined_splits dict for 'predefined' mode"
        def ids(list_ids): return [idx_map[sid] for sid in list_ids]

        train_ids = ids(predefined_splits['train'])
        val_ids   = ids(predefined_splits['val'])
        test_ids  = ids(predefined_splits.get('test', []))

        # IMPORTANT: dataset on CPU
        ds_train = RandomWindowSeqDataset([seqs[i] for i in train_ids], cfg, device=dataloader_device)
        ds_val   = RandomWindowSeqDataset([seqs[i] for i in val_ids],   cfg, device=dataloader_device)

        loader_train = _make_loader(ds_train, shuffle=shuffle_train, num_workers=num_workers)
        loader_val = DataLoader(
            ds_val,
            batch_size=cfg['train_params']['batch_size'],
            shuffle=False,
            num_workers=cfg['train_params']['batch_size'],
            pin_memory=pin_memory
        )

        if test_ids:
            ds_test = RandomWindowSeqDataset([seqs[i] for i in test_ids], cfg, device=dataloader_device)
            loader_test = DataLoader(
                ds_test,
                batch_size=cfg['train_params']['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory
            )
            return loader_train, loader_val, loader_test
        return loader_train, loader_val

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

# def build_dataloaders(seqs, cfg, device,
#                       split_type='rotary', predefined_splits=None):
#     """
#     Create train/val (and optional test) DataLoaders from seq list.

#     split_type:
#       - 'rotary': leave-one-out on seq_ids each epoch (returns a list of tuples)
#       - 'predefined': use explicit splits dict  

#     predefined_splits: {
#       'train': [seq_id, ...],
#       'val':   [seq_id, ...],
#       'test':  [seq_id, ...] (optional)
#     }

#     Returns:
#       - rotary: list of (holdout_seq_id, train_loader, val_loader)
#       - predefined: (train_loader, val_loader) or (train_loader, val_loader, test_loader)
#     """
#     idx_map = {s['seq_id']: i for i, s in enumerate(seqs)} # TODO: could add sorted idxs

#     # --- optionale Config-Parameter lesen ---
#     seed = cfg.get('train_params', {}).get('random_seed', None)
#     deterministic = cfg.get('train_params', {}).get('deterministic', False)
#     shuffle_train = cfg.get('train_params', {}).get('shuffle_train', True)
#     # setze Basis-Seed für Worker (top-level global)
#     set_worker_seed_base(seed)

#     g = None
#     worker_init_fn = None
#     if seed is not None:
#         os.environ["PYTHONHASHSEED"] = str(seed)
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#         g = torch.Generator()
#         g.manual_seed(seed)
#         wi_fn = worker_init_fn # Alias auf Top Level Funktion
#     else:
#         set_worker_seed_base(None)
#         wi_fn = None

#     if deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True

#     if split_type == 'rotary':
#         loaders = []
#         for holdout in idx_map:
#             val_idxs = [idx_map[holdout]]
#             train_idxs = [i for i in range(len(seqs)) if i not in val_idxs]

#             ds_train = RandomWindowSeqDataset([seqs[i] for i in train_idxs], cfg, device=device)
#             ds_val = RandomWindowSeqDataset([seqs[i] for i in val_idxs], cfg, device=device)

#             loader_train = DataLoader(
#                 ds_train,
#                 batch_size=cfg['train_params']['batch_size'],
#                 shuffle=shuffle_train,
#                 num_workers=cfg['train_params']['dataloader_num_workers'],
#                 multiprocessing_context=torch.multiprocessing.get_context('spawn'),
#                 generator=g,                    # NEU
#                 worker_init_fn=wi_fn,  # Neu, Alias
#                 persistent_workers=(cfg['train_params']['dataloader_num_workers'] > 0),
#                 drop_last=True
#             )
#             loader_val = DataLoader(
#                 ds_val,
#                 batch_size=cfg['train_params']['batch_size'],
#                 shuffle=False,
#                 num_workers=0
#             )
#             loaders.append((holdout, loader_train, loader_val))
#         return loaders

#     elif split_type == 'predefined':
#         assert predefined_splits is not None, "Need predefined_splits dict for 'predefined' mode"
#         def ids(list_ids): return [idx_map[sid] for sid in list_ids]

#         train_ids = ids(predefined_splits['train'])
#         val_ids   = ids(predefined_splits['val'])
#         test_ids  = ids(predefined_splits.get('test', []))

#         ds_train = RandomWindowSeqDataset([seqs[i] for i in train_ids], cfg, device=device)
#         ds_val   = RandomWindowSeqDataset([seqs[i] for i in val_ids], cfg, device=device)

#         loader_train = DataLoader(
#             ds_train,
#             batch_size=cfg['train_params']['batch_size'],
#             shuffle=shuffle_train,
#             num_workers=cfg['train_params']['dataloader_num_workers'],
#             multiprocessing_context=torch.multiprocessing.get_context('spawn'),
#             generator=g,                    # NEU
#             worker_init_fn=wi_fn,  # NEU, Alias
#             persistent_workers=(cfg['train_params']['dataloader_num_workers'] > 0),
#             drop_last=True
#         )
#         loader_val = DataLoader(
#             ds_val,
#             batch_size=cfg['train_params']['batch_size'],
#             shuffle=False,
#             num_workers=0
#         )

#         if test_ids:
#             ds_test = RandomWindowSeqDataset([seqs[i] for i in test_ids], cfg, device=device)
#             loader_test = DataLoader(
#                 ds_test,
#                 batch_size=cfg['train_params']['batch_size'],
#                 shuffle=False,
#                 num_workers=0
#             )
#             return loader_train, loader_val, loader_test
#         return loader_train, loader_val

#     else:
#         raise ValueError(f"Unknown split_type: {split_type}")