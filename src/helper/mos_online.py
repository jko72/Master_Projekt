"""Raw-point MOS targets and LMNet frame input (no image target warping)."""
from pathlib import Path
import numpy as np
import torch
from helper.mos_label_helper import map_semantickitti_to_mos
from helper.mos_projection import project_features, project_targets
from helper.mos_input import normalize_feature_tensor


def point_labels(sample):
    path = Path(sample['seq_dir']) / 'labels' / (sample['frame_stem'] + '.label')
    raw = np.fromfile(path, dtype=np.uint32)
    return map_semantickitti_to_mos(raw & 0xFFFF)


def is_test(sample):
    return sample['seq_id'] in {str(i).zfill(2) for i in range(11, 22)}


def lmnet_frame(dataset, sample):
    points = np.fromfile(sample['scan_path'], dtype=np.float32).reshape(-1, 4)
    labels = None if is_test(sample) else point_labels(sample)
    if labels is not None and len(labels) != len(points):
        raise ValueError('Point/label length mismatch')
    scans, labels = dataset.augmentation.raw_scans([points], labels)
    feature, mask, projection = project_features(scans[0], dataset.H, dataset.W, dataset.theta_range, True)
    y = None if labels is None else torch.from_numpy(project_targets(labels, projection))
    features, masks, y = dataset.augmentation.range_views([feature], [mask], y)
    feature, mask = features[0], masks[0]
    feature = normalize_feature_tensor(feature, mask, dataset.normalization)
    mode = dataset.input_mode
    names = ['x', 'y', 'z', 'range', 'intensity']
    if mode.startswith('range_xyz_intensity'):
        pass
    elif mode.startswith('range_xyz') and 'normal' not in mode:
        feature, names = feature[:4], names[:4]
    elif mode in {'range', 'range_residual'}:
        feature, names = feature[3:4], ['range']
    elif mode == 'residual':
        feature, names = feature[:0], []
    else:
        raise ValueError(f'Unsupported LMNet input mode: {mode}')
    if 'residual' in mode:
        from helper.mos_cache import check_cache
        residuals = []
        for off in dataset.residual_offsets:
            folder = dataset.cfg['mos_data_params']['residual_folder_template'].format(offset=off)
            directory = Path(sample['seq_dir']) / folder
            check_cache(directory, dataset.cfg, 'residual', sample['seq_dir'], off)
            residuals.append(torch.from_numpy(dataset._load_residual_image(str(directory / (sample['frame_stem']+'.npy')))))
            names.append(f'residual_{off}')
        feature = torch.cat([feature, torch.stack(residuals)])
    # Like the author parser, mask ALL channels by current projection occupancy.
    # The precomputed residual values themselves are never recomputed/warped.
    feature = feature.masked_fill(~mask.bool(), 0)
    if not torch.isfinite(feature).all():
        raise ValueError(f'Non-finite LMNet features: {sample["scan_path"]}')
    return feature, y, dict(sample, channel_names=names, normalization=dataset.normalization['mode'])
