"""Versioned geometric MOS caches, independent of loss/feature normalization."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from helper.mos_projection import projection_mode, REFERENCE_COMMIT


@lru_cache(maxsize=128)
def _file_hash(path, mtime, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_manifest(cfg, kind, seq_dir, offset=None):
    mp = cfg['model_params']
    source = {}
    for name in ('poses.txt', 'calib.txt'):
        path = Path(seq_dir) / name
        stat = path.stat() if path.exists() else None
        source[name] = _file_hash(str(path.resolve()), stat.st_mtime_ns, stat.st_size) if stat else None
    spec = dict(version='lmnet_v1', reference_commit=REFERENCE_COMMIT, kind=kind,
                height=int(mp['grid_height']), width=int(mp['grid_width']),
                fov_up=float(mp.get('FOV_UP', 3)), fov_down=float(mp.get('FOV_DOWN', -25)),
                projection_mode=projection_mode(cfg),
                filter=dict(min_range=2. if kind == 'residual' else 0.,
                            max_range=50. if kind == 'residual' else None,
                            finite=True, strict=True, before_collision=True),
                mapping='semantickitti_mos_static0_moving1_ignore-1_v1',
                offset=offset, residual_normalized=bool(cfg.get('residual_params', {}).get('normalize', True)) if kind == 'residual' else None,
                pose_calibration_source=source, pose_convention='inv(current_velodyne_pose)@past_velodyne_pose')
    spec['fingerprint'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def check_cache(folder, cfg, kind, seq_dir, offset=None, create=False):
    if projection_mode(cfg) != 'lmnet':
        return
    folder = Path(folder)
    if 'lmnet_v1' not in folder.name:
        raise ValueError(f'LMNet requires a separate lmnet_v1 cache folder: {folder}')
    expected = cache_manifest(cfg, kind, seq_dir, offset)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f'Incompatible MOS cache manifest: {manifest}')
    elif create:
        if folder.exists() and any(folder.glob('*.npy')):
            raise ValueError(f'Cannot adopt unversioned existing cache: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=2) + '\n')
    else:
        raise FileNotFoundError(f'Missing MOS cache manifest: {manifest}; generate the versioned cache first')
