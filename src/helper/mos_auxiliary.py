"""Optional task supervision and validated semantic bottleneck model options."""
import math
from numbers import Real

import torch
import torch.distributed as dist

from helper.mos_lovasz_loss import MOSCrossEntropyLovasz, loss_settings

MOVABLE_CONTENT = (0.9331685037721359, 0.047646460740965804)
MOVABLE_WEIGHTS = (1.07047068699, 20.55648005566)


def normalized_semantic_model_params(params):
    """Copy model params with explicit legacy defaults for resume comparisons."""
    result = dict(params or {})
    result.setdefault('movable_branch', 'shared_decoder')
    result.setdefault('semantic_fusion', 'none')
    return result


def semantic_model_settings(cfg):
    """Central validation for builder, trainer and saved-config reconstruction.

    `fusion` remains the existing temporal-fusion option. These two independent
    options select the movable feature source and semantic MOS feedback only.
    Legacy configurations add no modules and require no auxiliary supervision.
    """
    params = normalized_semantic_model_params(cfg.get('mos_model_params'))
    branch, fusion = params['movable_branch'], params['semantic_fusion']
    if branch not in ('shared_decoder', 'current_bottleneck'):
        raise ValueError('mos_model_params.movable_branch must be shared_decoder or current_bottleneck')
    if fusion not in ('none', 'bottleneck_gate'):
        raise ValueError('mos_model_params.semantic_fusion must be none or bottleneck_gate')
    if fusion == 'bottleneck_gate' and branch != 'current_bottleneck':
        raise ValueError('semantic_fusion=bottleneck_gate requires movable_branch=current_bottleneck')
    if branch == 'current_bottleneck':
        auxiliary = validate_auxiliary_config(cfg)
        if (params.get('name') != 'salsanext_temporal_e2e' or
                int(params.get('num_classes', 2)) != 2 or not auxiliary['enabled'] or
                auxiliary['task'] != 'movable'):
            raise ValueError('current_bottleneck requires binary salsanext_temporal_e2e and enabled movable auxiliary supervision')
        if params.get('fusion', 'bottleneck_only') != 'bottleneck_only':
            raise ValueError('Temporal fusion must remain bottleneck_only')
    return dict(movable_branch=branch, semantic_fusion=fusion)


def auxiliary_settings(value=None):
    """Validate every supplied key, including disabled configurations."""
    def mapping(item, keys, path):
        if not isinstance(item, dict) or set(item) - set(keys):
            raise ValueError(f'Invalid/unknown {path} settings')
        return item

    value = mapping({} if value is None else value,
                    ('enabled', 'task', 'loss_weight', 'movable', 'residual', 'inference'), 'mos_auxiliary')
    enabled = value.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('mos_auxiliary.enabled must be boolean')
    task = value.get('task', 'movable')
    if task not in ('movable', 'residual'):
        raise ValueError('mos_auxiliary.task must be movable or residual')
    if ('movable' in value and task != 'movable') or ('residual' in value and task != 'residual'):
        raise ValueError('Only one auxiliary task block may be supplied')
    weight = value.get('loss_weight', 0.3 if task == 'residual' else 1.0)
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError('mos_auxiliary.loss_weight must be finite and nonnegative')
    movable = mapping(value.get('movable', {}), ('class_weights', 'loss', 'ignore_index'), 'movable')
    weights = movable.get('class_weights', list(MOVABLE_WEIGHTS))
    if not isinstance(weights, (list, tuple)) or len(weights) != 2 or any(
            isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('movable.class_weights requires two positive finite numbers')
    ignore = movable.get('ignore_index', -1)
    if isinstance(ignore, bool) or ignore != -1:
        raise ValueError('movable.ignore_index must be -1')
    loss = loss_settings(movable.get('loss', {'type': 'ce_lovasz'}))
    if loss['type'] != 'ce_lovasz' or loss['lovasz_weight'] != 1.0:
        raise ValueError('movable loss requires CE + Lovasz with lovasz_weight=1')
    inference = mapping(value.get('inference', {}), ('use_auxiliary_output',), 'inference')
    if inference.get('use_auxiliary_output', False) is not False:
        raise ValueError('MOS inference always uses decoder.logits; auxiliary output is forbidden')
    if task == 'residual':
        defaults = dict(offset=8, min_range=2.0, max_range=50.0, epsilon=1e-6,
                        normalization='relative', loss='smooth_l1', beta=0.1,
                        ignore_invalid_pairs=True)
        residual = mapping(value.get('residual', {}), defaults, 'residual')
        spec = {**defaults, **residual}
        if isinstance(spec['offset'], bool) or not isinstance(spec['offset'], int) or spec['offset'] <= 0:
            raise ValueError('residual.offset must be a positive integer')
        for key in ('min_range', 'max_range', 'epsilon', 'beta'):
            n = spec[key]
            if isinstance(n, bool) or not isinstance(n, Real) or not math.isfinite(n) or n <= 0:
                raise ValueError(f'residual.{key} must be positive and finite')
            spec[key] = float(n)
        if spec['min_range'] >= spec['max_range']:
            raise ValueError('residual requires min_range < max_range')
        if spec['normalization'] != 'relative' or spec['loss'] != 'smooth_l1' or spec['ignore_invalid_pairs'] is not True:
            raise ValueError('Residual requires relative, smooth_l1 and ignore_invalid_pairs=true')
        return dict(enabled=enabled, task=task, loss_weight=float(weight), residual=spec,
                    inference=dict(use_auxiliary_output=False))
    return dict(enabled=enabled, task='movable', loss_weight=float(weight),
                movable=dict(class_weights=list(weights), loss=loss, ignore_index=-1),
                inference=dict(use_auxiliary_output=False))


def validate_auxiliary_config(cfg):
    settings = auxiliary_settings(cfg.get('mos_auxiliary'))
    if settings['enabled'] and settings['task'] == 'residual':
        from helper.mos_projection import projection_mode
        data = cfg.get('mos_data_params', {})
        if data.get('previous_offsets') != [settings['residual']['offset']]:
            raise ValueError('Residual requires exactly one previous_offsets entry matching residual.offset')
        if projection_mode(cfg) != 'lmnet':
            raise ValueError('Residual auxiliary requires projection_mode=lmnet')
        from helper.mos_augmentation import augmentation_settings
        aug = augmentation_settings(cfg.get('mos_augmentation'), data.get('input_mode', 'range_xyz_intensity'))
        if aug['enabled'] and aug['historical_point_dropout']['enabled']:
            raise ValueError('Residual targets require clean history: historical_point_dropout is unsupported')
    return settings


def attach_auxiliary_criterion(criterion, cfg, device):
    settings = validate_auxiliary_config(cfg)
    if settings['enabled']:
        criterion.auxiliary_loss_weight = settings['loss_weight']
        if settings['task'] == 'residual':
            from helper.mos_residual_auxiliary import MaskedResidualSmoothL1
            criterion.residual_criterion = MaskedResidualSmoothL1(settings['residual']['beta'])
            criterion.residual_settings = settings['residual']
            return criterion
        spec = settings['movable']
        loss = spec['loss']
        criterion.movable_criterion = MOSCrossEntropyLovasz(
            torch.tensor(spec['class_weights'], dtype=torch.float32, device=device),
            loss['lovasz_weight'], loss['lovasz_classes'], loss['lovasz_per_image'])
    return criterion


def corrected_task_loss(criterion, components, target):
    """Global weighted CE per microbatch; rank-local Lovasz, as in baseline.

    Invoke for each task separately and only during training. Empty local or
    global supervision stays differentiably zero and participates in DDP.
    """
    ce = components['loss_ce']
    if dist.is_available() and dist.is_initialized():
        valid = target != -1
        local_weight = criterion.weight[target[valid]].sum() if criterion.weight is not None else valid.sum()
        global_weight = local_weight.detach().clone()
        dist.all_reduce(global_weight)
        ce = ce * local_weight * dist.get_world_size() / global_weight.clamp_min(1e-12)
    return ce + getattr(criterion, 'lovasz_weight', 0.) * components['loss_lovasz']
