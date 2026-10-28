"""Five-channel masked Smooth-L1 reconstruction; no auxiliary tasks."""
import math
import torch
from torch.nn import functional as F


def reconstruction_settings(value=None):
    defaults = dict(xyz_weight=.5, range_weight=1., intensity_weight=.25, beta=1.)
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('Unsupported mae_params.loss settings')
    spec = {**defaults, **value}
    for key, val in spec.items():
        if isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val) or val < 0:
            raise ValueError(f'mae_params.loss.{key} must be a nonnegative finite number')
    if spec['beta'] <= 0 or not any(spec[k] > 0 for k in defaults if k.endswith('_weight')):
        raise ValueError('Smooth-L1 beta and at least one reconstruction weight must be positive')
    return spec


def reconstruction_loss(pred, target, valid, mask, settings=None):
    spec = reconstruction_settings(settings)
    if pred.ndim != 4 or pred.shape[1] != 5 or target.shape != pred.shape:
        raise ValueError('Reconstruction expects matching [B,5,H,W] prediction and target')
    expected = pred[:, 0].shape
    if valid.shape != expected or mask.shape != expected:
        raise ValueError('valid and mask must be [B,H,W]')
    selected = valid.bool() & mask.bool()
    losses, sums, counts = {}, {}, {}
    # Promote AMP outputs: geometry and reductions remain in FP32.
    pred, target = pred.float(), target.float()
    for name, channels in [('xyz', slice(0, 3)), ('range', slice(3, 4)), ('intensity', slice(4, 5))]:
        p, t = pred[:, channels], target[:, channels]
        finite = torch.isfinite(p).all(1) & torch.isfinite(t).all(1)
        take = (selected & finite)[:, None].expand_as(p)
        # Sanitize BEFORE loss arithmetic; excluded NaNs must have zero gradients.
        p, t = torch.where(take, p, 0.), torch.where(take, t, 0.)
        total = F.smooth_l1_loss(p, t, reduction='sum', beta=spec['beta'])
        count = take.sum()
        sums[name], counts[name] = total, count
        losses['loss_'+name] = total / count.clamp_min(1)
    loss = sum(spec[name+'_weight'] * losses['loss_'+name] for name in sums)
    stats = {**losses, 'loss_total': loss,
             'valid_pixels': valid.bool().sum(), 'pixels': valid.numel(),
             'masked_valid_pixels': selected.sum(), 'sums': sums, 'counts': counts}
    return loss, stats
