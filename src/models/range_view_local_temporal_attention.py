"""Local historical-to-current attention on cylindrical range-view features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def local_candidates(history: torch.Tensor, radius: int) -> torch.Tensor:
    """[B,T,C,H,W] -> [B,H*W,T*(2r+1)^2,C], oldest frame first.

    Azimuth is circular. Elevation padding is zero and must be masked using
    an unfolded occupancy map before attention.
    """
    b, t, c, h, w = history.shape
    flat = history.reshape(b * t, c, h, w)
    if radius:
        flat = F.pad(flat, (radius, radius, 0, 0), mode="circular")
        flat = F.pad(flat, (0, 0, radius, radius))
    k = 2 * radius + 1
    windows = F.unfold(flat, kernel_size=k).reshape(b, t, c, k * k, h * w)
    return windows.permute(0, 4, 1, 3, 2).reshape(b, h * w, t * k * k, c)


class LocalHistoricalAttention(nn.Module):
    """Shared comparison projection plus local historical K/V and no-match token."""

    def __init__(self, channels: int = 256, heads: int = 8, radius: int = 1,
                 compare_channels: int = 64):
        super().__init__()
        if channels % heads or channels <= 0 or heads <= 0 or radius < 0:
            raise ValueError("channels must divide heads; radius must be nonnegative")
        self.channels, self.heads, self.radius = channels, heads, radius
        self.compare_channels = compare_channels
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.compare = nn.Conv2d(channels, compare_channels, 1)
        self.relative_key = nn.Linear(3, channels, bias=False)
        self.null_key = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.null_value = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.null_compare = nn.Parameter(torch.zeros(1, 1, 1, compare_channels))
        self.change = nn.Sequential(
            nn.Conv2d(channels + compare_channels + 6, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, current: torch.Tensor, history: torch.Tensor,
                current_valid: torch.Tensor, history_valid: torch.Tensor,
                history_dt: torch.Tensor, return_diagnostics: bool = False):
        b, c, h, w = current.shape
        if history.ndim != 5 or history.shape[0] != b or history.shape[2:] != (c, h, w):
            raise ValueError("history must be [B,T-1,C,H,W] matching current")
        t = history.shape[1]
        if t < 1 or current_valid.shape != (b, 1, h, w) or history_valid.shape != (b, t, 1, h, w):
            raise ValueError("invalid temporal validity shapes")
        if history_dt.shape != (b, t) or not torch.isfinite(history_dt).all() or not (history_dt < 0).all():
            raise ValueError("history_dt must be finite negative seconds [B,T-1]")
        n, r, heads, dim = h * w, self.radius, self.heads, c // self.heads
        k2 = (2 * r + 1) ** 2
        query = self.query(current).flatten(2).transpose(1, 2)  # B,N,C
        key = local_candidates(self.key(history.reshape(b * t, c, h, w)).reshape(b, t, c, h, w), r)
        value = local_candidates(self.value(history.reshape(b * t, c, h, w)).reshape(b, t, c, h, w), r)
        cmp_hist = local_candidates(self.compare(history.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w), r)
        cmp_current = self.compare(current).flatten(2).transpose(1, 2)
        candidate_valid = local_candidates(history_valid.float(), r)[..., 0] > 0.5
        dy = torch.arange(-r, r + 1, device=current.device, dtype=current.dtype)
        dx = torch.arange(-r, r + 1, device=current.device, dtype=current.dtype)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")
        spatial = torch.stack((grid_y.flatten(), grid_x.flatten()), dim=-1).repeat(t, 1)
        relative = torch.cat((spatial[None].expand(b, -1, -1),
                              history_dt.to(current.dtype).repeat_interleave(k2, dim=1)[..., None]), dim=-1)
        key = key + self.relative_key(relative)[:, None]
        key = torch.cat((key, self.null_key.to(key.dtype).expand(b, n, -1, -1)), dim=2)
        value = torch.cat((value, self.null_value.to(value.dtype).expand(b, n, -1, -1)), dim=2)
        cmp_hist = torch.cat((cmp_hist, self.null_compare.to(cmp_hist.dtype).expand(b, n, -1, -1)), dim=2)
        qh = query.reshape(b, n, heads, dim)
        kh = key.reshape(b, n, t * k2 + 1, heads, dim)
        scores = (qh[:, :, None].float() * kh.float()).sum(-1) / math.sqrt(dim)
        key_mask = torch.cat((candidate_valid, torch.ones((b, n, 1), device=current.device, dtype=torch.bool)), dim=-1)
        scores = scores.masked_fill(~key_mask[..., None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2).to(value.dtype)
        context = (weights[..., None] * value.reshape(b, n, t * k2 + 1, heads, dim)).sum(2).reshape(b, n, c)
        mean_weights = weights.float().mean(-1)
        matched = (mean_weights[..., None] * cmp_hist.float()).sum(2)
        difference = cmp_current.float() - matched
        history_weights = mean_weights[..., :-1]
        null_weight = mean_weights[..., -1]
        best_match = history_weights.max(-1).values
        expected = (history_weights[..., None] * relative[:, None].float()).sum(2)
        entropy = -(mean_weights * mean_weights.clamp_min(1e-8).log()).sum(-1)
        stats = torch.cat((null_weight[..., None], best_match[..., None], expected,
                           entropy[..., None]), dim=-1)
        change_input = torch.cat((context.float(), difference, stats), dim=-1)
        delta = self.change(change_input.transpose(1, 2).reshape(b, -1, h, w).to(current.dtype))
        fused = current + self.gamma.to(current.dtype) * delta * current_valid.to(current.dtype)
        if not return_diagnostics:
            return fused
        names = ("null_weight", "best_match", "dy", "dx", "dt", "entropy")
        diagnostics = {name: stats[..., i].reshape(b, h, w).detach() for i, name in enumerate(names)}
        return fused, diagnostics
