"""FP32 masked regression and bounded-memory, mergeable residual diagnostics."""
import math
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class MaskedResidualSmoothL1(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = float(beta)

    def components(self, prediction, target, valid_mask):
        if prediction.ndim != 4 or prediction.shape[1] != 1 or prediction[:, 0].shape != target.shape:
            raise ValueError('Expected residual prediction [B,1,H,W] and target [B,H,W]')
        if valid_mask.dtype != torch.bool or valid_mask.shape != target.shape:
            raise ValueError('Expected bool residual valid_mask with target shape')
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            p, t = prediction[:, 0].float()[valid_mask], target.float()[valid_mask]
            # Index before subtraction: invalid NaN targets never enter the graph.
            total = F.smooth_l1_loss(p, t, beta=self.beta, reduction='sum')
            count = valid_mask.sum()
            mean = total / count.clamp_min(1)
        return dict(loss_sum=total, valid_count=count, loss_total=mean)

    def forward(self, prediction, target, valid_mask):
        return self.components(prediction, target, valid_mask)['loss_total']


class ResidualStatistics:
    """Global additive sums and histogram quantiles; no per-batch collectives.

    Target upper bound is max_range/min_range for positive filtered ranges.
    Values are never clipped. Quantiles use 8192 bins with documented precision.
    """
    names = ('loss_sum', 'count', 'pixels', 'absolute_error', 'squared_error',
             'target_sum', 'near_zero', 'current_pixels', 'history_pixels',
             'clean_range_valid_pixels', 'dropped_or_changed_winner_pixels')

    def __init__(self, settings, device):
        self.bins = 8192
        self.upper = settings['max_range'] / settings['min_range']
        self.sums = torch.zeros(len(self.names), dtype=torch.float64, device=device)
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prediction, target, valid_mask, components, stats=None):
        p, t = prediction[:, 0].float()[valid_mask], target.float()[valid_mask]
        error = p - t
        values = [components['loss_sum'].detach(), valid_mask.sum(), valid_mask.numel(),
                  error.abs().double().sum(), error.double().square().sum(), t.double().sum(),
                  (t <= 1e-6).sum()]
        stats = stats or {}
        values.extend(torch.as_tensor(stats.get(k, 0), device=self.sums.device).sum()
                      for k in self.names[7:])
        self.sums += torch.stack([torch.as_tensor(v, device=self.sums.device, dtype=torch.float64) for v in values])
        self.histogram += torch.histc(t, bins=self.bins, min=0, max=self.upper).double()

    def finish(self, mos_loss, loss_weight):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.sums)
            dist.all_reduce(self.histogram)
        values = dict(zip(self.names, self.sums.tolist()))
        count = values['count']
        denominator = max(count, 1)
        loss = values['loss_sum'] / denominator
        weighted = loss_weight * loss
        result = dict(aux_loss=loss, aux_smooth_l1=loss,
                      aux_mae=values['absolute_error']/denominator,
                      aux_rmse=(values['squared_error']/denominator)**0.5,
                      aux_valid_ratio=count/max(values['pixels'], 1), aux_valid_pixels=int(count),
                      aux_target_mean=values['target_sum']/denominator,
                      aux_target_near_zero_ratio=values['near_zero']/denominator,
                      aux_target_quantile_bin_width=self.upper/self.bins,
                      lambda_residual=loss_weight, aux_weighted_loss=weighted,
                      aux_to_mos_loss_ratio=weighted/max(mos_loss, 1e-12),
                      total_loss=mos_loss+weighted, multitask_loss_total=mos_loss+weighted)
        cumulative = self.histogram.cumsum(0)
        for name, q in (('p50', .5), ('p90', .9), ('p99', .99)):
            value = 0.
            if count:
                # Match linear sample-quantile interpolation between order stats.
                position = (count-1)*q
                ranks = torch.tensor([math.floor(position)+1, math.ceil(position)+1],
                                     dtype=cumulative.dtype, device=cumulative.device)
                lower, upper = torch.searchsorted(cumulative, ranks).clamp(max=self.bins-1).tolist()
                fraction = position-math.floor(position)
                value = ((1-fraction)*lower+fraction*upper)*self.upper/self.bins
            # Order-statistic bin lower endpoints: absolute error <= one bin width.
            result['aux_target_'+name] = value
        result.update({'aux_'+k: int(values[k]) for k in self.names[7:]})
        return result
