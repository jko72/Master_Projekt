"""Versioned full-network MAE metadata and strict head-only MOS transfer."""
from pathlib import Path
import warnings

import torch
from helper.mos_input import BASE_CHANNELS, normalization

FORMAT = 'temporal_e2e_masked_reconstruction_v1'
HEAD = {'decoder.logits.weight', 'decoder.logits.bias'}


def model_metadata(cfg):
    p = cfg['mos_model_params']
    defaults = dict(name='salsanext_temporal_e2e', in_channels=5, dropout=.1,
                    attention_heads=8, radius=1, compare_channels=64, fusion='bottleneck_only')
    return {key: p.get(key, val) for key, val in defaults.items()}


def input_metadata(cfg):
    data = cfg['mos_data_params']
    return dict(channel_names=list(BASE_CHANNELS), in_channels=5,
                input_mode=data['input_mode'], normalization=normalization(cfg, data['input_mode']),
                previous_offsets=sorted({int(v) for v in data['previous_offsets']}, reverse=True),
                scan_rate_hz=float(data.get('scan_rate_hz', 10.)), projection=dict(cfg['model_params']),
                model=model_metadata(cfg))


def validate_mae_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != FORMAT:
        raise ValueError('Expected a versioned Temporal E2E MAE reconstruction checkpoint')
    if not isinstance(checkpoint.get('config'), dict) or not isinstance(checkpoint.get('metadata'), dict):
        raise ValueError('MAE checkpoint lacks configuration/metadata')
    if checkpoint['metadata'] != input_metadata(checkpoint['config']):
        raise ValueError('MAE checkpoint metadata disagrees with its configuration')
    if checkpoint['metadata']['input_mode'] != 'range_xyz_intensity' or checkpoint['metadata']['model']['in_channels'] != 5:
        raise ValueError('MAE checkpoint must use five XYZRI input channels')
    if checkpoint['config']['mos_model_params'].get('num_classes') != 5:
        raise ValueError('MAE checkpoint must use five reconstruction outputs')
    if not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('MAE checkpoint lacks full model_state_dict')
    return checkpoint['metadata']


def load_mae_weights(model, path, cfg):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source_meta = validate_mae_checkpoint(checkpoint)
    target_meta = input_metadata(cfg)
    for key in ('channel_names', 'in_channels', 'input_mode', 'normalization', 'model'):
        if source_meta[key] != target_meta[key]:
            raise ValueError(f'MAE transfer incompatible {key}: {source_meta[key]} vs {target_meta[key]}')
    if source_meta['previous_offsets'] != target_meta['previous_offsets']:
        warnings.warn(f"MAE transfer offsets differ: {source_meta['previous_offsets']} -> {target_meta['previous_offsets']}", UserWarning)
    source, target = checkpoint['model_state_dict'], model.state_dict()
    if set(source) != set(target):
        raise ValueError(f'MAE transfer key mismatch; missing={sorted(set(target)-set(source))}, unexpected={sorted(set(source)-set(target))}')
    for key, tensor in source.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f'MAE transfer non-tensor state: {key}')
        expected = ((5, *target[key].shape[1:]) if key in HEAD else target[key].shape)
        if tuple(tensor.shape) != tuple(expected) or tensor.dtype != target[key].dtype:
            raise ValueError(f'MAE transfer shape/dtype mismatch: {key}, got {tensor.shape}/{tensor.dtype}, expected {expected}/{target[key].dtype}')
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f'MAE transfer non-finite state: {key}')
    if model.decoder.logits.out_channels != 2:
        raise ValueError('MOS transfer target must have two output channels')
    transfer = {key: tensor for key, tensor in source.items() if key not in HEAD}
    # Validate everything before mutating the target. Strictly retain only its new head.
    model.load_state_dict({**transfer, **{key: target[key] for key in HEAD}}, strict=True)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Full MOS fine-tuning requires all parameters trainable')
    report = dict(path=str(Path(path).resolve()), format=FORMAT, epoch=checkpoint['epoch'],
                  loaded_keys=len(transfer), excluded_keys=sorted(HEAD), metadata=source_meta)
    print(f"[MAE -> MOS] Loaded {len(transfer)} backbone tensors; new two-channel head; epoch={checkpoint['epoch']}", flush=True)
    return report
