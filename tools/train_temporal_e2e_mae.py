#!/usr/bin/env python3
"""Masked temporal reconstruction of XYZRI using the complete E2E MOS backbone."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import datetime
import math
import random
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import train_temporal_mos_e2e as mos
from helper.mos_eval_utils import select_sequences
from helper.mos_input import normalization
from helper.mos_augmentation import augmentation_settings
from helper.temporal_e2e_transfer import FORMAT, input_metadata, validate_mae_checkpoint, checkpoint_format
from helper.mos_projection import projection_mode
from temporal_e2e_mae_dataset import TemporalE2EMAEDataset, mask_settings, masked_input
from loss_temporal_e2e_mae import reconstruction_loss, reconstruction_settings


def read_config(path):
    path = Path(path)
    if not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    p, data = cfg['mos_model_params'], cfg['mos_data_params']
    if p.get('name') != 'salsanext_temporal_e2e' or p.get('in_channels') != 5 or p.get('num_classes') != 5:
        raise ValueError('MAE requires salsanext_temporal_e2e with five inputs and five outputs')
    if p.get('fusion', 'bottleneck_only') != 'bottleneck_only':
        raise ValueError('MAE uses bottleneck_only temporal fusion')
    for key, default in (('attention_heads', 8), ('radius', 1), ('compare_channels', 64)):
        value = p.get(key, default)
        if type(value) is not int or value < (0 if key == 'radius' else 1):
            raise ValueError(f'Invalid mos_model_params.{key}')
    if 256 % p.get('attention_heads', 8):
        raise ValueError('attention_heads must divide the 256 bottleneck channels')
    dropout = p.get('dropout', .1)
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('dropout must be finite and in [0,1)')
    if data.get('input_mode') != 'range_xyz_intensity':
        raise ValueError('MAE requires range_xyz_intensity')
    normalization(cfg, data['input_mode'])
    offsets = data.get('previous_offsets', [])
    if not offsets or any(type(v) is not int or v <= 0 for v in offsets):
        raise ValueError('previous_offsets must contain positive integer offsets')
    rate = data.get('scan_rate_hz', 10.)
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('scan_rate_hz must be positive and finite')
    mp = cfg['model_params']
    if any(type(mp[k]) is not int or mp[k] <= 0 or mp[k] % 16 for k in ('grid_height', 'grid_width')):
        raise ValueError('MAE image dimensions must be positive multiples of 16')
    mos.check_splits(data)
    cfg['mos_augmentation'] = augmentation_settings(cfg.get('mos_augmentation'), data['input_mode'])
    projection_mode(cfg)
    mae = cfg['mae_params']
    if set(mae) - {'mask', 'loss'}:
        raise ValueError('Only mask and reconstruction loss are supported in mae_params')
    mae['mask'] = mask_settings(mae.get('mask', {}))
    mae['loss'] = reconstruction_settings(mae.get('loss'))
    train = cfg['mae_train_params']
    defaults = dict(epochs=50, batch_size=8, num_workers=2, learning_rate=.0005,
                    learning_rate_min=.000005, weight_decay=.0001, num_warmup_epochs=2,
                    clip_grad_norm=5., seed=42, amp=True, amp_dtype='bfloat16', deterministic=False,
                    optimizer='adamw', scheduler='warmup_cosine')
    cfg['mae_train_params'] = train = {**defaults, **train}
    accumulation = train.get('gradient_accumulation', 1)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('mae_train_params.gradient_accumulation must be a positive integer')
    for key in ('epochs', 'batch_size', 'num_workers', 'num_warmup_epochs', 'seed'):
        if type(train[key]) is not int or train[key] < (1 if key in ('epochs', 'batch_size') else 0):
            raise ValueError(f'mae_train_params.{key} has an invalid integer value')
    for key in ('learning_rate', 'learning_rate_min', 'weight_decay', 'clip_grad_norm'):
        val = train[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            raise ValueError(f'mae_train_params.{key} must be finite and nonnegative')
    if not 0 <= train['learning_rate_min'] <= train['learning_rate'] or train['learning_rate'] <= 0:
        raise ValueError('Require 0 <= learning_rate_min <= learning_rate and learning_rate > 0')
    if train['optimizer'] != 'adamw' or train['scheduler'] != 'warmup_cosine':
        raise ValueError('MAE requires AdamW and warmup_cosine')
    if train['amp_dtype'] not in {'bfloat16', 'float16'} or any(type(train[k]) is not bool for k in ('amp', 'deterministic')):
        raise ValueError('Invalid AMP or deterministic settings')
    return path.resolve(), cfg


def amp_settings(cfg, device):
    # Share the established dtype/device checks without changing MOS settings.
    return mos.amp_settings({'mos_train_params': cfg['mae_train_params']}, device)


def make_loader(cfg, split, device):
    if split not in {'train', 'val'}:
        raise ValueError('MAE cannot load the test split')
    t, data = cfg['mae_train_params'], cfg['mos_data_params']
    sequences = select_sequences(mos.make_sequences(cfg['dataset_path']), data[split+'_sequences'], split)
    dataset = TemporalE2EMAEDataset(sequences, cfg, split)
    if not len(dataset):
        raise ValueError(f'No usable MAE {split} samples')
    sampler = None
    if mos.distributed():
        if split == 'train':
            sampler = DistributedSampler(dataset, shuffle=True, seed=t['seed'])
        else:
            dataset = Subset(dataset, range(dist.get_rank(), len(dataset), dist.get_world_size()))
    rank = dist.get_rank() if mos.distributed() else 0
    generator = torch.Generator().manual_seed(t['seed'] + rank)
    return DataLoader(dataset, batch_size=t['batch_size'], num_workers=t['num_workers'],
                      sampler=sampler, shuffle=split == 'train' and sampler is None,
                      pin_memory=device.type == 'cuda', generator=generator)


def set_epoch(loader, epoch):
    dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    dataset.set_epoch(epoch)
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def make_scheduler(optimizer, train):
    warm, epochs = train['num_warmup_epochs'], train['epochs']
    floor = train['learning_rate_min'] / train['learning_rate']
    def multiplier(epoch):
        if warm and epoch < warm:
            return (epoch+1)/warm
        progress = min(1., max(0., (epoch-warm)/max(1, epochs-warm-1)))
        return floor + (1-floor)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def phase(model, loader, cfg, device, optimizer=None, scaler=None, max_batches=None):
    if optimizer is not None and cfg['mae_train_params'].get('gradient_accumulation', 1) > 1:
        return accumulated_phase(model, loader, cfg, device, optimizer, scaler, max_batches)
    training = optimizer is not None
    model.train(training)
    if not training:
        mos.sync_buffers(model)
    forward = model if training else mos.unwrap_model(model)
    amp, dtype = amp_settings(cfg, device)
    spec = cfg['mae_params']['loss']
    # Component numerators/counts allow exact global pixel means in DDP.
    totals = torch.zeros(9, device=device, dtype=torch.float64)
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x, valid, dt, mask, target = [batch[k].to(device, non_blocking=True)
                                     for k in ('x', 'valid', 'dt', 'mask', 'target_current')]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                # Original validity is preserved. Unmasked x/target never reach forward.
                pred = forward(masked_input(x, mask), valid, dt)
                loss, stats = reconstruction_loss(pred, target, valid[:, -1], mask, spec)
            if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite MAE prediction/loss at batch {index}')
            if training:
                backward = loss
                if mos.distributed():
                    counts = torch.stack([stats['counts'][n] for n in ('xyz', 'range', 'intensity')]).detach()
                    dist.all_reduce(counts)
                    backward = sum(spec[n+'_weight'] * stats['sums'][n] * dist.get_world_size() /
                                   counts[i].clamp_min(1) for i, n in enumerate(('xyz', 'range', 'intensity')))
                scaler.scale(backward).backward()
                scaler.unscale_(optimizer)
                clip = cfg['mae_train_params']['clip_grad_norm']
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float('inf'),
                                               error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
        values = [stats['sums'][n].detach() for n in ('xyz', 'range', 'intensity')]
        values += [stats['counts'][n].detach() for n in ('xyz', 'range', 'intensity')]
        values += [stats['valid_pixels'], stats['masked_valid_pixels'], stats['pixels']]
        totals += torch.as_tensor([float(v) for v in values], dtype=torch.float64, device=device)
        if mos.primary() and (index == 0 or (index+1) % 20 == 0):
            print(f"[E2E-MAE {'train' if training else 'val'}] batch={index+1} loss={float(loss.detach()):.6f}", flush=True)
    if mos.distributed():
        dist.all_reduce(totals)
    if totals[8] == 0:
        raise ValueError('MAE phase produced no samples')
    if totals[7] == 0:
        raise ValueError('MAE phase contains no valid masked reconstruction targets')
    results = {'loss_'+n: float(totals[i]/totals[i+3].clamp_min(1))
               for i, n in enumerate(('xyz', 'range', 'intensity'))}
    results['loss_total'] = sum(spec[n+'_weight']*results['loss_'+n] for n in ('xyz', 'range', 'intensity'))
    results.update(valid_ratio=float(totals[6]/totals[8]), masked_valid_ratio=float(totals[7]/totals[6].clamp_min(1)))
    return results


def accumulated_phase(model, loader, cfg, device, optimizer, scaler, max_batches=None):
    """Exact masked-pixel objective across each CPU-buffered accumulation group.

    Only one microbatch lives on the GPU. BatchNorm still uses microbatch
    statistics; accumulation does not emulate a larger physical BatchNorm batch.
    """
    model.train()
    amp, dtype = amp_settings(cfg, device)
    train, spec = cfg['mae_train_params'], cfg['mae_params']['loss']
    accumulation = train['gradient_accumulation']
    iterator = iter(loader) if max_batches is None else islice(loader, max_batches)
    totals = torch.zeros(9, device=device, dtype=torch.float64)
    names = ('xyz', 'range', 'intensity')
    channel_slices = (slice(0, 3), slice(3, 4), slice(4, 5))
    steps = skipped = samples = batch_index = 0
    world = dist.get_world_size() if mos.distributed() else 1
    while True:
        group = list(islice(iterator, accumulation))
        if not group:
            break
        # Denominators are known before forward: predictions must be finite.
        counts = torch.zeros(3, dtype=torch.float64, device=device)
        for batch in group:
            selected = batch['valid'][:, -1].bool() & batch['mask'].bool()
            for j, channels in enumerate(channel_slices):
                target = batch['target_current'][:, channels]
                count = ((selected & torch.isfinite(target).all(1)).sum() * target.shape[1])
                counts[j] += count.to(device)
        if mos.distributed():
            dist.all_reduce(counts)
        if not (counts > 0).any():
            raise ValueError('Accumulation group has no valid masked targets')
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(group):
            synchronize = index == len(group)-1
            context = model.no_sync() if isinstance(model, mos.DDP) and not synchronize else nullcontext()
            x, valid, dt, mask, target = [batch[k].to(device, non_blocking=True)
                                         for k in ('x', 'valid', 'dt', 'mask', 'target_current')]
            with context:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    pred = model(masked_input(x, mask), valid, dt)
                    loss, stats = reconstruction_loss(pred, target, valid[:, -1], mask, spec)
                if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite MAE prediction/loss at batch {batch_index}')
                objective = sum(spec[n+'_weight'] * stats['sums'][n] * world /
                                counts[j].clamp_min(1) for j, n in enumerate(names))
                scaler.scale(objective).backward()
            values = [stats['sums'][n].detach() for n in names]
            values += [stats['counts'][n].detach() for n in names]
            values += [stats['valid_pixels'], stats['masked_valid_pixels'], stats['pixels']]
            totals += torch.tensor([float(v) for v in values], dtype=torch.float64, device=device)
            samples += x.shape[0]
            batch_index += 1
            if mos.primary() and (batch_index == 1 or batch_index % 20 == 0):
                print(f'[E2E-MAE train] batch={batch_index} loss={float(loss.detach()):.6f} accumulation={accumulation}', flush=True)
        scaler.unscale_(optimizer)
        clip = train['clip_grad_norm']
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float('inf'),
                                       error_if_nonfinite=not scaler.is_enabled())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
        else:
            steps += 1
    if mos.distributed():
        dist.all_reduce(totals)
        sample_tensor = torch.tensor(samples, device=device)
        dist.all_reduce(sample_tensor)
        samples = int(sample_tensor)
    if totals[8] == 0 or totals[7] == 0:
        raise ValueError('MAE phase contains no valid masked reconstruction targets')
    results = {'loss_'+n: float(totals[j]/totals[j+3].clamp_min(1)) for j, n in enumerate(names)}
    results['loss_total'] = sum(spec[n+'_weight']*results['loss_'+n] for n in names)
    results.update(valid_ratio=float(totals[6]/totals[8]), masked_valid_ratio=float(totals[7]/totals[6].clamp_min(1)),
                   optimizer_steps=steps, amp_skipped_steps=skipped, samples=samples)
    return results


def capture_rng(loaders):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loaders=[loader.generator.get_state() for loader in loaders])


def restore_rng(state, loaders):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    for loader, rng in zip(loaders, state['loaders']):
        loader.generator.set_state(rng)


def checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, best, rng):
    raw = mos.unwrap_model(model)
    return dict(format=checkpoint_format(cfg), model_state_dict=raw.state_dict(),
                encoder_state_dict=raw.encoder.state_dict(), temporal_state_dict=raw.temporal.state_dict(),
                decoder_state_dict=raw.decoder.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
                epoch=epoch, best_val_loss=best, config=deepcopy(cfg), metadata=input_metadata(cfg),
                rng_by_rank=rng)


def resume_training(path, model, optimizer, scheduler, scaler, cfg, loaders, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_mae_checkpoint(checkpoint)
    old = checkpoint['config']
    for key in ('dataset_path', 'mos_model_params', 'model_params', 'mos_data_params', 'mae_params', 'mae_train_params', 'mos_augmentation'):
        if old[key] != cfg[key]:
            raise ValueError(f'Resume configuration mismatch: {key}; use the original run config')
    world = dist.get_world_size() if mos.distributed() else 1
    if len(checkpoint['rng_by_rank']) != world:
        raise ValueError('Resume requires the original DDP world size')
    mos.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Optimizer.load_state_dict casts moment tensors to the parameters device.
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    rank = dist.get_rank() if mos.distributed() else 0
    restore_rng(checkpoint['rng_by_rank'][rank], loaders)
    return checkpoint['epoch'], checkpoint['best_val_loss']


def history_diagnostics(model, batch, cfg, device):
    """Measure changes in masked predictions; no improvement is assumed."""
    model.eval()
    x, valid, dt, mask, target = [batch[k].to(device) for k in ('x', 'valid', 'dt', 'mask', 'target_current')]
    x = masked_input(x, mask)
    with torch.inference_mode():
        reference = model(x, valid, dt)
        results = {}
        for mode in ('zero', 'copy_current'):
            variant, occupancy = x.clone(), valid.clone()
            if mode == 'zero':
                variant[:, :-1] = 0
                occupancy[:, :-1] = False
            else:
                # Copy only the MASKED current input; never leak target pixels.
                variant[:, :-1] = x[:, -1:]
                occupancy[:, :-1] = valid[:, -1:]
            pred = model(variant, occupancy, dt)
            selected = (valid[:, -1] & mask)[:, None].expand_as(pred)
            delta = (pred-reference).abs()[selected]
            loss, _ = reconstruction_loss(pred, target, valid[:, -1], mask, cfg['mae_params']['loss'])
            results[mode] = dict(masked_prediction_mean_abs_delta=float(delta.mean()) if delta.numel() else 0., loss=float(loss))
    return results


def train(args, device):
    source, cfg = read_config(args.cfg_path)
    t = cfg['mae_train_params']
    for key in ('epochs', 'batch_size', 'num_workers', 'gradient_accumulation'):
        override = getattr(args, key, None)
        if override is not None:
            if override < (0 if key == 'num_workers' else 1):
                raise ValueError(f'Invalid --{key}')
            t[key] = override
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError('--max_batches must be positive')
    rank = dist.get_rank() if mos.distributed() else 0
    world = dist.get_world_size() if mos.distributed() else 1
    cfg['run_metadata'] = dict(task='temporal_e2e_masked_reconstruction', world_size=world,
                               batch_size_per_gpu=t['batch_size'],
                               effective_global_batch_size=t['batch_size']*world*t.get('gradient_accumulation', 1),
                               cli_args=vars(args))
    if 'gradient_accumulation' in t:
        cfg['run_metadata']['gradient_accumulation'] = t['gradient_accumulation']
    if projection_mode(cfg) == 'lmnet':
        cfg['run_metadata'].update(input_protocol='lmnet_v1', reconstruction_units='standardized_xyzri',
            batchnorm_batch='microbatch', mask_patch=list(cfg['mae_params']['mask'][k] for k in ('patch_h','patch_w')))
    mos.seed_all(t['seed'] + rank, t['deterministic'])
    amp, dtype = amp_settings(cfg, device)
    model = mos.wrap_distributed(mos.build_mos_model(cfg).to(device), device)
    if not all(p.requires_grad for p in model.parameters()):
        raise AssertionError('All MAE parameters must be trainable')
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scheduler = make_scheduler(optimizer, t)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    if args.dry_run:
        b, frames = 1, len(set(cfg['mos_data_params']['previous_offsets']))+1
        h, w = cfg['model_params']['grid_height'], cfg['model_params']['grid_width']
        generator = torch.Generator().manual_seed(t['seed'])
        x = torch.randn(b, frames, 5, h, w, generator=generator)
        valid = torch.ones(b, frames, h, w, dtype=torch.bool)
        from temporal_e2e_mae_dataset import patch_mask
        mask = patch_mask(valid[0, -1], cfg['mae_params']['mask'], generator)[None]
        dt = torch.tensor([[-v/cfg['mos_data_params']['scan_rate_hz'] for v in sorted(set(cfg['mos_data_params']['previous_offsets']), reverse=True)]+[0.]])
        batch = dict(x=x, valid=valid, dt=dt, mask=mask, target_current=x[:, -1].clone())
        result = phase(model, [batch], cfg, device, optimizer, scaler)
        health = mos.gradient_health(model)
        if any(health.values()):
            raise AssertionError(f'MAE gradient check failed: {health}')
        print(f'[E2E-MAE dry-run] {result}; gradients={health}', flush=True)
        return
    loaders = [make_loader(cfg, split, device) for split in ('train', 'val')]
    start, best = 0, float('inf')
    if args.resume:
        start, best = resume_training(args.resume, model, optimizer, scheduler, scaler, cfg, loaders, device)
    writer, run = None, None
    if mos.primary():
        log = cfg['mae_log_params']
        root = Path(args.log_root or log['log_root'])
        run = root / (log['run_name']+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        (run/'checkpoints').mkdir(parents=True)
        (run/'csv').mkdir()
        shutil.copy2(source, run/'config_source.yaml')
        (run/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        if args.resume:
            previous_best = Path(args.resume).resolve().parent/'best_val_loss.pt'
            if previous_best.is_file():
                shutil.copy2(previous_best, run/'checkpoints/best_val_loss.pt')
        if log.get('use_tensorboard', True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(run/'tensorboard'))
            except ImportError:
                print('[E2E-MAE] TensorBoard unavailable; CSV remains enabled', flush=True)
        print(f'[E2E-MAE] run={run} starting_epoch={start+1} amp={amp}/{dtype}', flush=True)
    try:
        for epoch in range(start, t['epochs']):
            for loader in loaders:
                set_epoch(loader, epoch)
            lr = optimizer.param_groups[0]['lr']
            training = phase(model, loaders[0], cfg, device, optimizer, scaler, args.max_batches)
            validation = phase(model, loaders[1], cfg, device, max_batches=args.max_batches)
            scheduler.step()
            improved = validation['loss_total'] < best
            best = min(best, validation['loss_total'])
            local_rng = capture_rng(loaders)
            rng = [local_rng]
            if mos.distributed():
                rng = [None]*dist.get_world_size()
                dist.all_gather_object(rng, local_rng)
            if mos.primary():
                row = dict(epoch=epoch+1, learning_rate=lr)
                for split, metrics in (('train', training), ('val', validation)):
                    row.update({split+'/'+key: val for key, val in metrics.items()})
                    if writer:
                        for key, val in metrics.items():
                            writer.add_scalar(split+'/'+key, val, epoch+1)
                with (run/'csv/epochs.csv').open('a', newline='') as handle:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                    if epoch == start:
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                checkpoint = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch+1, best, rng)
                torch.save(checkpoint, run/'checkpoints/last.pt')
                if improved:
                    torch.save(checkpoint, run/'checkpoints/best_val_loss.pt')
                print(f"[E2E-MAE] epoch={epoch+1} train={training['loss_total']:.6f} val={validation['loss_total']:.6f} best={best:.6f}", flush=True)
        if args.history_diagnostic and not mos.distributed():
            print('[E2E-MAE history diagnostic]', history_diagnostics(model, next(iter(loaders[1])), cfg, device), flush=True)
    finally:
        if writer:
            writer.close()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_path', default='src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_p2.yaml')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_root')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--gradient_accumulation', type=int, help='Optimizer accumulation steps; defaults to config or 1')
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--history-diagnostic', action='store_true')
    args = parser.parse_args()
    device = mos.setup_device(args.device)
    try:
        train(args, device)
    finally:
        if mos.distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
