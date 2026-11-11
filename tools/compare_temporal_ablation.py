#!/usr/bin/env python3
"""Compare three matched inference interventions using aggregate confusion counts."""
import argparse
import csv
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from helper.mos_benchmark_eval import point_metrics

MODES = ('none', 'mask_history', 'bypass_temporal')
PROTOCOL_KEYS = ('checkpoint', 'checkpoint_sha256', 'checkpoint_epoch', 'checkpoint_model_config',
                 'eval_sequences', 'dataset_root', 'n_scans', 'moving_threshold', 'projection',
                 'normalization', 'input_mode', 'history_init', 'history_reference_offsets',
                 'pose_source', 'gt_mapping', 'postprocessing', 'partial_smoke_test', 'pipeline_git_commit')
CONFUSION_KEYS = ('tp_moving', 'fp_moving', 'fn_moving', 'tn_static')


def read_rows(directory, name):
    with (directory / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def integer_counts(row, keys):
    try:
        result = {key: int(row[key]) for key in keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Missing or invalid integer counts: {keys}') from exc
    if any(v < 0 for v in result.values()):
        raise ValueError('Confusion counts and sample totals must be nonnegative')
    return result


def validate_metadata(metadata):
    first = metadata[0]
    if any(not isinstance(meta, dict) for meta in metadata):
        raise ValueError('Evaluation metadata must be mappings')
    for key in PROTOCOL_KEYS:
        if any(key not in meta or meta[key] != first.get(key) for meta in metadata):
            raise ValueError(f'Non-comparable evaluations: {key}')
    if not first['checkpoint_sha256'] or first['checkpoint_epoch'] is None:
        raise ValueError('Missing checkpoint SHA256 or epoch')
    if first['checkpoint_model_config']['mos_model_params']['name'] != 'salsanext_temporal_e2e':
        raise ValueError('Temporal comparison requires SalsaNextTemporalE2E')
    if first['moving_threshold'] != 0.5:
        raise ValueError('Temporal comparison requires moving_threshold=0.5')
    for meta, mode in zip(metadata, MODES):
        expected = dict(mode=mode, checkpoint_unchanged=True, historical_inputs_masked=mode == 'mask_history',
                        temporal_module_executed=mode != 'bypass_temporal', semantic_gate_unchanged=True)
        if meta.get('temporal_ablation') != expected:
            raise ValueError(f'Expected temporal_ablation mode={mode} with unchanged semantic gate/checkpoint')
        gate = meta.get('semantic_gate_ablation', {})
        fusion = meta['checkpoint_model_config']['mos_model_params'].get('semantic_fusion', 'none')
        expected_gate = dict(supported=fusion == 'bottleneck_gate', checkpoint_fusion=fusion,
                             gate_enabled=fusion == 'bottleneck_gate', disable_semantic_gate=False,
                             mode='runtime_forward_bypass', checkpoint_unchanged=True)
        if gate != expected_gate:
            raise ValueError('Additional semantic gate ablations are not allowed')
        if meta.get('threshold_sweep', {}).get('enabled') or meta.get('movable_diagnostics', {}).get('enabled'):
            raise ValueError('Comparison requires fixed-threshold evaluation without additional diagnostics')


def aggregate_counts(directory, metadata, level):
    rows = read_rows(directory, f'{level}_metrics.csv')
    if level == 'pixel':
        rows = [row for row in rows if row.get('sequence') == 'ALL']
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one aggregate {level}_metrics row')
    extra = ('n_gt_valid', 'n_points', 'n_scans') if level == 'point' else ('correct', 'valid_pixels', 'n_scans')
    counts = integer_counts(rows[0], (*CONFUSION_KEYS, *extra))
    valid_key = 'n_gt_valid' if level == 'point' else 'valid_pixels'
    if sum(counts[k] for k in CONFUSION_KEYS) != counts[valid_key]:
        raise ValueError(f'Invalid {level} aggregate confusion counts')
    if counts['n_scans'] != metadata['n_scans']:
        raise ValueError(f'Invalid {level} scan totals')
    if level == 'point' and counts['n_points'] < counts['n_gt_valid']:
        raise ValueError('Invalid point/GT totals')
    if level == 'pixel' and counts['correct'] != counts['tp_moving'] + counts['tn_static']:
        raise ValueError('Invalid pixel correct count')
    # Recompute every reported metric, ignoring stored IoU/precision/recall/F1.
    rates = point_metrics(counts)
    return counts, {f'{level}_{key.removeprefix("point_")}': rates[key]
                    for key in ('point_moving_iou', 'point_moving_precision', 'point_moving_recall', 'point_moving_f1')}


def compare(normal_dir, mask_history_dir, bypass_temporal_dir, output_dir):
    directories = [Path(d) for d in (normal_dir, mask_history_dir, bypass_temporal_dir)]
    metadata = [yaml.safe_load((d/'benchmark_eval_config.yaml').read_text()) for d in directories]
    validate_metadata(metadata)
    scan_keys = ('sequence', 'frame', 'n_points', 'history_available')
    scans = []
    for directory in directories:
        records = read_rows(directory, 'projection_coverage.csv')
        if any(any(k not in row for k in scan_keys) for row in records):
            raise ValueError('Missing scan identity/count/history metadata')
        scan_list = [tuple(row[k] for k in scan_keys) for row in records]
        if len({(row['sequence'], row['frame']) for row in records}) != len(records):
            raise ValueError('Duplicate scan identities')
        scans.append(scan_list)
    if any(s != scans[0] for s in scans) or len(scans[0]) != metadata[0]['n_scans']:
        raise ValueError('Non-comparable scan identities/counts/history availability')
    try:
        n_points = sum(int(scan[2]) for scan in scans[0])
        if any(int(scan[2]) < 0 for scan in scans[0]):
            raise ValueError('Negative scan point count')
    except ValueError as exc:
        raise ValueError('Invalid per-scan point counts') from exc
    pixel_available = [(d/'pixel_metrics.csv').is_file() for d in directories]
    if any(pixel_available) and not all(pixel_available):
        raise ValueError('Pixel metrics must be present in all three evaluations or none')
    rows = []
    for level in ('point', 'pixel') if all(pixel_available) else ('point',):
        values = [aggregate_counts(d, m, level) for d, m in zip(directories, metadata)]
        for counts, _ in values:
            if level == 'point' and counts['n_points'] != n_points:
                raise ValueError('Aggregate n_points differs from scan coverage')
        first = values[0][0]
        for counts, _ in values[1:]:
            if (counts['tp_moving'] + counts['fn_moving'] != first['tp_moving'] + first['fn_moving']
                    or counts['tn_static'] + counts['fp_moving'] != first['tn_static'] + first['fp_moving']):
                raise ValueError(f'Non-comparable {level} ground-truth class totals')
        metrics = list(values[0][1]) + [f'{level}_{k}' for k in first]
        for key in metrics:
            rate = key in values[0][1]
            numbers = [rates[key] if rate else counts[key.removeprefix(level+'_')]
                       for counts, rates in values]
            normal, masked, bypassed = numbers
            rows.append(dict(metric=key, normal=normal, mask_history=masked, bypass_temporal=bypassed,
                             delta_mask_history_minus_normal=masked-normal,
                             delta_bypass_temporal_minus_normal=bypassed-normal,
                             delta_mask_history_percentage_points=100*(masked-normal) if rate else None,
                             delta_bypass_temporal_percentage_points=100*(bypassed-normal) if rate else None))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir/'temporal_ablation_comparison.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir/'temporal_ablation_comparison.yaml').open('x', encoding='utf-8') as handle:
        yaml.safe_dump(dict(normal_dir=str(directories[0].resolve()), mask_history_dir=str(directories[1].resolve()),
                            bypass_temporal_dir=str(directories[2].resolve()),
                            protocol={k: metadata[0][k] for k in PROTOCOL_KEYS},
                            temporal_attention_gamma=metadata[0].get('temporal_attention_gamma'), metrics=rows),
                       handle, sort_keys=False)
    for row in rows:
        suffix = (f" delta(mask-normal)={row['delta_mask_history_percentage_points']:+.6f} pp"
                  f" delta(bypass-normal)={row['delta_bypass_temporal_percentage_points']:+.6f} pp"
                  if row['delta_mask_history_percentage_points'] is not None else '')
        print(f"{row['metric']}: normal={row['normal']:.9g} mask={row['mask_history']:.9g} "
              f"bypass={row['bypass_temporal']:.9g}{suffix}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-dir', required=True, type=Path)
    parser.add_argument('--mask-history-dir', required=True, type=Path)
    parser.add_argument('--bypass-temporal-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    compare(args.normal_dir, args.mask_history_dir, args.bypass_temporal_dir, args.output_dir)


if __name__ == '__main__':
    main()
