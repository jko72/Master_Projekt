"""Configuration and epoch counters for train-only temporal offset sampling."""
from collections.abc import Mapping
import math
from numbers import Real

import torch
import torch.distributed as dist


def validate_temporal_offset_sampling(cfg):
    """Validate without modifying configs (especially legacy checkpoint configs)."""
    data = cfg.get('mos_data_params', {})
    if 'temporal_offset_sampling' not in data:
        return {'enabled': False}
    settings = data['temporal_offset_sampling']
    prefix = 'mos_data_params.temporal_offset_sampling'
    if not isinstance(settings, Mapping):
        raise ValueError(f'{prefix} must be a mapping')
    unknown = set(settings) - {'enabled', 'offsets', 'probabilities', 'train_only'}
    if unknown:
        raise ValueError(f'{prefix}: unknown parameters {sorted(unknown, key=str)}')
    if type(settings.get('enabled', False)) is not bool:
        raise ValueError(f'{prefix}.enabled must be a Boolean')
    if settings.get('train_only', True) is not True:
        raise ValueError(f'{prefix}.train_only must be true')
    # A minimal explicit disable is accepted, with no defaults written to cfg.
    if not settings.get('enabled', False) and not ({'offsets', 'probabilities'} & settings.keys()):
        return dict(settings, enabled=False)
    offsets = settings.get('offsets')
    if not isinstance(offsets, list) or not offsets:
        raise ValueError(f'{prefix}.offsets must be a nonempty list')
    if any(type(off) is not int or off <= 0 for off in offsets):
        raise ValueError(f'{prefix}.offsets must contain positive integers')
    if len(set(offsets)) != len(offsets):
        raise ValueError(f'{prefix}.offsets must not contain duplicates')
    probabilities = settings.get('probabilities')
    if not isinstance(probabilities, list) or len(probabilities) != len(offsets):
        raise ValueError(f'{prefix}.probabilities must be a list matching offsets')
    if any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or p < 0
           for p in probabilities):
        raise ValueError(f'{prefix}.probabilities must be finite numbers >= 0')
    if not any(p > 0 for p in probabilities) or not math.isclose(sum(probabilities), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError(f'{prefix}.probabilities must sum to 1 with at least one positive value')
    if settings.get('enabled', False):
        reference = data.get('previous_offsets', [3])
        if (not isinstance(reference, list) or len(reference) != 1
                or type(reference[0]) is not int or reference[0] <= 0):
            raise ValueError(f'{prefix} requires exactly one positive integer in previous_offsets')
        if data.get('projection_mode', 'legacy') != 'lmnet':
            raise ValueError(f'{prefix} requires the LMNet temporal E2E pipeline')
        if cfg.get('mos_model_params', {}).get('name') != 'salsanext_temporal_e2e':
            raise ValueError(f'{prefix} requires salsanext_temporal_e2e')
        if (cfg.get('mos_auxiliary') or {}).get('task') == 'residual':
            raise ValueError(f'{prefix} is incompatible with residual auxiliary supervision')
    return dict(settings, enabled=settings.get('enabled', False))


class TemporalOffsetStatistics:
    """CPU batch counters; one reduction at phase end, never from workers."""
    def __init__(self, dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        settings = getattr(dataset, 'offset_sampling', {})
        self.enabled = bool(settings.get('enabled', False) and dataset.split == 'train')
        self.offsets = sorted(set(settings['offsets']) | set(dataset.offsets)) if self.enabled else []
        self.counts = torch.zeros(len(self.offsets) + 2, dtype=torch.int64)

    def update(self, meta):
        selected = meta['sampled_previous_offset']
        for j, off in enumerate(self.offsets):
            self.counts[j] += (selected == off).sum().cpu()
        self.counts[-2] += (~meta['history_available'][0].bool()).sum().cpu()
        self.counts[-1] += selected.numel()

    def finish(self, device):
        totals = self.counts.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        values = totals.cpu().tolist()
        total = values[-1]
        result = {'missing_history_count': values[-2], 'offset_sample_count': total}
        for off, count in zip(self.offsets, values):
            result[f'offset_{off}_count'] = count
            result[f'offset_{off}_ratio'] = count / total if total else 0.
        return result
