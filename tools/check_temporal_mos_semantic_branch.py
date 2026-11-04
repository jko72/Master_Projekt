#!/usr/bin/env python3
"""Prepared compute-node acceptance checks; never run automatically at training startup.

Run explicitly in the existing HPC container after allocation. Synthetic checks
use 32x64 inputs; the separate two-rank training smokes use real 64x2048 data.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from helper.mos_auxiliary import auxiliary_settings, semantic_model_settings
from mos_models import build_mos_model
from train_temporal_mos_e2e import validate_resume_model_params
from evaluate_mos_benchmark import load_adapter


def forward_contract(model, device):
    model.eval()
    channels = model.in_channels
    values = (torch.randn(1, 2, channels, 32, 64, device=device),
              torch.ones(1, 2, 32, 64, dtype=torch.bool, device=device),
              torch.tensor([[-.8, 0.]], device=device))
    with torch.inference_mode():
        mos = model(*values)
        mos_aux, movable = model(*values, return_aux=True)
        torch.testing.assert_close(mos, mos_aux, rtol=0, atol=0)
        if mos.shape != (1, 2, 32, 64) or movable.shape != mos.shape:
            raise AssertionError('Incorrect MOS/movable output shapes')
        if not torch.isfinite(mos).all() or not torch.isfinite(movable).all():
            raise AssertionError('Non-finite checkpoint outputs')
    return values, mos


def check_legacy(path, reference, device):
    cfg = yaml.safe_load(reference.read_text())
    if semantic_model_settings(cfg) != dict(movable_branch='shared_decoder', semantic_fusion='none'):
        raise ValueError('The supplied reference must select the legacy shared-decoder path')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    saved = payload.get('config', payload.get('cfg'))
    if not isinstance(saved, dict):
        raise ValueError('Legacy checkpoint lacks configuration')
    validate_resume_model_params(saved, cfg)
    aux = auxiliary_settings(saved.get('mos_auxiliary'))
    if not aux['enabled'] or aux['task'] != 'movable':
        raise ValueError('Expected a legacy movable checkpoint')
    model = build_mos_model(cfg)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device)
    forward_contract(model, device)
    print(json.dumps(dict(check='real_legacy_checkpoint', checkpoint=str(path),
                          strict=True, model_options=semantic_model_settings(cfg),
                          reference=str(reference), parameter_count=sum(p.numel() for p in model.parameters()))), flush=True)


def check_saved(path, device, args):
    model, cfg, payload = load_adapter(path, device)
    options = semantic_model_settings(cfg)
    if options['movable_branch'] != 'current_bottleneck':
        raise ValueError('--checkpoint expects one of the new compact-branch checkpoints')
    if args.expect_fusion and options['semantic_fusion'] != args.expect_fusion:
        raise ValueError('Checkpoint selects the other semantic fusion')
    if payload.get('model_options') != options:
        raise ValueError('Checkpoint model_options metadata disagrees with its config')
    metadata = cfg.get('run_metadata', {})
    if metadata.get('model_options') != options:
        raise ValueError('Run model_options metadata disagrees with its config')
    if args.expect_world_size is not None and metadata.get('world_size') != args.expect_world_size:
        raise ValueError('Unexpected training world_size')
    values, expected = forward_contract(model, device)
    with tempfile.TemporaryDirectory(prefix='semantic-roundtrip-') as directory:
        temporary = Path(directory)/'roundtrip.pt'
        torch.save(dict(config=cfg, model_state_dict=model.state_dict(), model_options=options), temporary)
        restored, _, _ = load_adapter(temporary, device)
        with torch.inference_mode():
            torch.testing.assert_close(restored(*values), expected, rtol=0, atol=0)
    report = dict(check='new_checkpoint_roundtrip', checkpoint=str(path), model_options=options,
                  strict=True, world_size=metadata.get('world_size'),
                  parameter_count=sum(p.numel() for p in model.parameters()))
    if args.expect_min_updates is not None:
        with (path.parent.parent/'csv/epochs.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise AssertionError('No completed training/validation epoch in CSV')
        row = rows[-1]
        updates = int(row['train_optimizer_updates'])
        if updates < args.expect_min_updates:
            raise AssertionError(f'Only {updates} optimizer updates, expected at least {args.expect_min_updates}')
        for phase in ('train', 'val'):
            if int(row[f'{phase}_samples']) <= 0:
                raise AssertionError(f'No {phase} samples')
            for name in ('loss_ce', 'loss_lovasz', 'movable_loss_ce', 'movable_loss_lovasz', 'multitask_loss_total'):
                if not math.isfinite(float(row[f'{phase}_{name}'])):
                    raise AssertionError(f'Non-finite {phase}_{name}')
        report['optimizer_updates'] = updates
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run-suite', action='store_true')
    parser.add_argument('--two-gpu-ddp', action='store_true', help='Enable the prepared two-rank synthetic NCCL test')
    parser.add_argument('--reference-config', type=Path,
                        default=ROOT/'src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml')
    parser.add_argument('--legacy-checkpoint', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expect-fusion', choices=('none', 'bottleneck_gate'))
    parser.add_argument('--expect-world-size', type=int)
    parser.add_argument('--expect-min-updates', type=int)
    args = parser.parse_args()
    if not (args.run_suite or args.legacy_checkpoint or args.checkpoint):
        parser.error('Request --run-suite, --legacy-checkpoint or --checkpoint')
    if args.two_gpu_ddp and not args.run_suite:
        parser.error('--two-gpu-ddp requires --run-suite')
    if args.two_gpu_ddp and (not args.device.startswith('cuda') or torch.cuda.device_count() < 2):
        parser.error('--two-gpu-ddp requires two allocated visible CUDA GPUs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; run on an allocated compute node')
    if args.run_suite:
        os.environ['MOS_CHECK_DEVICE'] = str(device)
        os.environ['MOS_CHECK_DDP'] = '1' if args.two_gpu_ddp else '0'
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_temporal_mos_semantic_branch.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    if args.legacy_checkpoint:
        check_legacy(args.legacy_checkpoint, args.reference_config, device)
    if args.checkpoint:
        check_saved(args.checkpoint, device, args)


if __name__ == '__main__':
    main()
