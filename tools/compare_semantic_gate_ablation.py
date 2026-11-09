#!/usr/bin/env python3
"""Compare paired runtime gate evaluations, using aggregate confusion counts."""
import argparse
import csv
from pathlib import Path
import sys
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from helper.mos_benchmark_eval import point_metrics


def read_rows(directory, name):
    with (Path(directory) / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def compare(gate_on_dir, gate_off_dir, output_dir):
    directories = [Path(gate_on_dir), Path(gate_off_dir)]
    metadata = [yaml.safe_load((d / 'benchmark_eval_config.yaml').read_text()) for d in directories]
    first, second = metadata
    required = ('checkpoint', 'checkpoint_sha256', 'checkpoint_epoch', 'checkpoint_model_config',
                'eval_sequences', 'dataset_root', 'n_scans', 'moving_threshold', 'projection',
                'normalization', 'input_mode', 'history_init', 'pose_source', 'gt_mapping',
                'postprocessing', 'partial_smoke_test', 'pipeline_git_commit')
    for key in required:
        if key not in first or key not in second or first[key] != second[key]:
            raise ValueError(f'Non-comparable evaluations: {key}')
    for meta, disabled in zip(metadata, (False, True)):
        gate = meta.get('semantic_gate_ablation', {})
        expected = dict(supported=True, checkpoint_fusion='bottleneck_gate',
                        gate_enabled=not disabled, disable_semantic_gate=disabled,
                        mode='runtime_forward_bypass', checkpoint_unchanged=True)
        if gate != expected:
            raise ValueError('Expected gate ON followed by gate OFF runtime_forward_bypass')
        if meta.get('movable_diagnostics', {}).get('enabled') or meta.get('threshold_sweep', {}).get('enabled'):
            raise ValueError('Gate comparison requires fixed-threshold MOS evaluation without movable diagnostics')
    # Compare scan identity and original point counts, not timing/coverage differences.
    scan_keys = ('sequence', 'frame', 'n_points', 'history_available')
    scans = [[tuple(row.get(k) for k in scan_keys) for row in read_rows(d, 'projection_coverage.csv')]
             for d in directories]
    if scans[0] != scans[1] or len(scans[0]) != int(first['n_scans']):
        raise ValueError('Non-comparable scan identities/counts')
    values = []
    count_keys = ('tp_moving', 'fp_moving', 'fn_moving', 'tn_static', 'n_gt_valid', 'n_points', 'n_scans')
    for directory, meta in zip(directories, metadata):
        rows = read_rows(directory, 'point_metrics.csv')
        if len(rows) != 1:
            raise ValueError('Expected exactly one aggregated point_metrics row')
        counts = {key: int(rows[0][key]) for key in count_keys}
        if any(v < 0 for v in counts.values()) or sum(counts[k] for k in count_keys[:4]) != counts['n_gt_valid']:
            raise ValueError('Invalid aggregate confusion counts')
        if counts['n_points'] < counts['n_gt_valid'] or counts['n_scans'] != meta['n_scans']:
            raise ValueError('Invalid point/scan totals')
        values.append({**counts, **point_metrics(counts)})
    for key in ('n_points', 'n_gt_valid', 'n_scans'):
        if values[0][key] != values[1][key]:
            raise ValueError(f'Non-comparable {key}')
    metrics = ('point_moving_iou', 'point_moving_precision', 'point_moving_recall',
               'point_moving_f1', *count_keys)
    rows = []
    for key in metrics:
        on, off = (v[key] for v in values)
        if not math.isfinite(on) or not math.isfinite(off):
            raise ValueError(f'Non-finite metric: {key}')
        rows.append(dict(metric=key, gate_on=on, gate_off=off, delta_on_minus_off=on-off,
                         delta_percentage_points=100*(on-off) if key.startswith('point_') else None))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'gate_ablation_comparison.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'gate_ablation_comparison.yaml').open('x') as handle:
        yaml.safe_dump(dict(gate_on_dir=str(directories[0].resolve()), gate_off_dir=str(directories[1].resolve()),
                            protocol={k: first[k] for k in required}, metrics=rows), handle, sort_keys=False)
    for row in rows:
        print(f"{row['metric']}: ON={row['gate_on']} OFF={row['gate_off']} delta={row['delta_on_minus_off']:+g}"
              + (f" ({row['delta_percentage_points']:+.6f} pp)" if row['delta_percentage_points'] is not None else ''))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-on-dir', required=True, type=Path)
    parser.add_argument('--gate-off-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    compare(args.gate_on_dir, args.gate_off_dir, args.output_dir)


if __name__ == '__main__':
    main()
