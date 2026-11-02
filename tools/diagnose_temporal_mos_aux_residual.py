#!/usr/bin/env python3
"""Save aligned metric range images, residual target and mask for one real scan."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_temporal_mos_e2e import read_config
from helper.dataloader_helper import make_sequences
from helper.mos_projection import lmnet_projection
from helper.residual_helper import compute_residual_target, projection_ranges
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_path', default='src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml')
    parser.add_argument('--dataset-root')
    parser.add_argument('--sequence', default='08')
    parser.add_argument('--frame', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, cfg = read_config(args.cfg_path)
    if cfg.get('mos_auxiliary', {}).get('task') != 'residual' or not cfg['mos_auxiliary']['enabled']:
        parser.error('An enabled residual configuration is required')
    if args.dataset_root:
        cfg['dataset_path'] = args.dataset_root
    seqs = [s for s in make_sequences(cfg['dataset_path']) if str(s['seq_id']).zfill(2) == args.sequence.zfill(2)]
    if not seqs:
        parser.error('Requested sequence is missing')
    dataset = TemporalMOSE2EDataset(seqs, cfg, 'val')
    if not 0 <= args.frame < len(dataset):
        parser.error('Frame index is outside the sequence')
    sample = dataset.samples[args.frame]
    _, _, maps, _, _, available, details = dataset._project_lmnet_temporal(sample, return_clean=True)
    clean = [lmnet_projection(p, dataset.H, dataset.W, cfg['model_params']['FOV_UP'], cfg['model_params']['FOV_DOWN'])
             for p in details['clean_scans']]
    target, valid, stats = compute_residual_target(clean[-1], clean[0], current_input=maps[-1],
        input_original_ids=details['current_original_ids'], history_available=available[0],
        settings=cfg['mos_auxiliary']['residual'], return_diagnostics=True)
    values = target[valid]
    stats.update(sequence=args.sequence, frame=args.frame, history_available=available[0],
                 residual_valid_pixels=int(valid.sum()), residual_valid_ratio=float(valid.mean()),
                 near_zero_ratio=float((values <= 1e-6).mean()) if values.size else 0.,
                 target_mean=float(values.mean()) if values.size else 0.,
                 target_quantiles=dict(zip(('p50','p90','p99'), np.quantile(values,[.5,.9,.99]).tolist())) if values.size else {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing an existing diagnostic artifact.
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, current_range=projection_ranges(clean[-1]),
            historical_range_aligned=projection_ranges(clean[0]), current_input_range=projection_ranges(maps[-1]),
            residual_target=target, residual_valid=valid, current_clean_indices=clean[-1].indices,
            current_input_indices=maps[-1].indices, surviving_original_ids=details['current_original_ids'])
    print(json.dumps(stats, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
