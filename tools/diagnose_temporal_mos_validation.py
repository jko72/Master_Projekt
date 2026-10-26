#!/usr/bin/env python3
"""Read every temporal MOS validation sample without DataLoader workers or a model.

The tool is intended to isolate sequence-08 data access and range-view
projection.  It flushes the sample identifier *before* loading it so that an
abrupt process exit still leaves the last attempted frame in the Slurm log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helper.dataloader_helper import make_sequences
from helper.mos_eval_utils import select_sequences
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from train_temporal_mos_e2e import check_splits, read_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based validation-sample index at which to begin.")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of samples to inspect.")
    return parser.parse_args()


def main():
    args = parse_args()
    _, cfg = read_config(args.cfg_path)
    check_splits(cfg["mos_data_params"])
    data = cfg["mos_data_params"]
    sequences = select_sequences(make_sequences(cfg["dataset_path"]), data["val_sequences"], "val")
    dataset = TemporalMOSE2EDataset(sequences, cfg, split="val")

    start = args.start_index
    if not 0 <= start < len(dataset):
        raise ValueError(f"start-index must be in [0, {len(dataset) - 1}], got {start}")
    stop = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    offsets = dataset.offsets
    print(f"[validation-data] samples={len(dataset)} inspecting=[{start}, {stop}) "
          f"offsets={offsets} workers=0 model=none", flush=True)

    for index in range(start, stop):
        sample = dataset.samples[index]
        sequence = dataset.sequence_map[sample["seq_id"]]
        frame_index = sample["frame_index"]
        history = [dataset._extract_scan_path(sequence["paths"][frame_index - offset])
                   for offset in offsets]
        print(f"[validation-data] loading index={index} seq={sample['seq_id']} "
              f"frame={sample['frame_stem']} current={sample['scan_path']} "
              f"history={history}", flush=True)
        x, valid, dt, target, meta = dataset[index]
        if not (torch.isfinite(x).all() and torch.isfinite(dt).all()):
            raise FloatingPointError(f"Non-finite tensor at index={index}: {meta}")
        if not (valid[-1] & (target != -1)).any():
            raise ValueError(f"No supervised current pixels at index={index}: {meta}")
        print(f"[validation-data] ok index={index} x={tuple(x.shape)} "
              f"valid_current={int(valid[-1].sum())}", flush=True)

    print(f"[validation-data] passed {stop - start} samples", flush=True)


if __name__ == "__main__":
    main()
