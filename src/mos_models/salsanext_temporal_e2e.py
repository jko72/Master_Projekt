"""Shared temporal E2E architecture for MOS and masked five-channel reconstruction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.range_view_local_temporal_attention import LocalHistoricalAttention
from .salsanext_parts import SalsaNextDecoder, SalsaNextEncoder


class CurrentBottleneckMovableHead(nn.Module):
    """Current-only semantic detail at the existing H/4, W/4 encoder skip.

    Only two raw logits are interpolated to full resolution. The shared
    semantic adapter lives on the parent model so the gate consumes the exact
    same S_t, without running this small decoder during MOS-only inference.
    """

    def __init__(self, skip_channels):
        super().__init__()
        self.skip_projection = nn.Conv2d(skip_channels, 32, kernel_size=1)
        self.refine = nn.Sequential(nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1), nn.GELU())
        self.logits = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, semantic, current_skip, output_size):
        semantic = F.interpolate(semantic, size=current_skip.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat((semantic, self.skip_projection(current_skip)), dim=1)
        logits = self.logits(self.refine(features))
        return F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)


class SalsaNextTemporalE2E(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, dropout=0.1,
                 attention_heads=8, radius=1, compare_channels=64, auxiliary_movable=False,
                 auxiliary_task=None, movable_branch='shared_decoder', semantic_fusion='none'):
        super().__init__()
        if in_channels not in {4, 5} or num_classes not in {2, 5} or (num_classes == 5 and in_channels != 5):
            raise ValueError("Temporal E2E requires XYZR/XYZRI MOS (2 outputs) or XYZRI reconstruction (5 outputs)")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder = SalsaNextEncoder(in_channels, dropout)
        self.temporal = LocalHistoricalAttention(256, attention_heads, radius, compare_channels)
        self.decoder = SalsaNextDecoder(num_classes, dropout)
        if auxiliary_movable and auxiliary_task not in (None, 'movable'):
            raise ValueError('Only one auxiliary task can be enabled')
        self.auxiliary_task = auxiliary_task or ('movable' if auxiliary_movable else None)
        if self.auxiliary_task not in (None, 'movable', 'residual'):
            raise ValueError('Unknown auxiliary task')
        self.auxiliary_movable = self.auxiliary_task == 'movable'
        # Also protect direct construction; the normal entry uses the same
        # central validator in build_mos_model before allocating parameters.
        from helper.mos_auxiliary import semantic_model_settings
        options = semantic_model_settings(dict(
            mos_model_params=dict(name='salsanext_temporal_e2e', num_classes=num_classes,
                                  movable_branch=movable_branch, semantic_fusion=semantic_fusion),
            mos_auxiliary=dict(enabled=self.auxiliary_task is not None,
                               task=self.auxiliary_task or 'movable')))
        self.movable_branch = options['movable_branch']
        self.semantic_fusion = options['semantic_fusion']
        if self.auxiliary_task is not None:
            if num_classes != 2:
                raise ValueError('Movable supervision requires the binary MOS model')
            # Construct on CPU, preserving the surrounding CPU/CUDA RNG streams.
            # The original backbone and subsequent augmentation draws stay identical.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1729)
                if self.movable_branch == 'current_bottleneck':
                    channels = self.encoder.res_block5.conv1.out_channels
                    skip_channels = self.encoder.res_block3.conv1.out_channels
                    self.semantic_adapter = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=1), nn.GELU())
                    self.movable_head = CurrentBottleneckMovableHead(skip_channels)
                elif self.auxiliary_movable:
                    self.movable_head = nn.Conv2d(32, 2, kernel_size=1)
                else:
                    self.residual_head = nn.Conv2d(32, 1, kernel_size=1)
        if self.semantic_fusion == 'bottleneck_gate':
            # Separate stream: adapter/head weights and the caller's RNG stream
            # match the no-gate ablation exactly. Zero P starts as identity.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1730)
                self.semantic_gate = nn.Conv2d(64, self.encoder.res_block5.conv1.out_channels, kernel_size=1)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.zeros_(self.semantic_gate.bias)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, dt: torch.Tensor,
                return_diagnostics: bool = False, return_aux: bool = False):
        if return_aux and self.auxiliary_task is None:
            raise ValueError('return_aux requires enabled auxiliary supervision')
        if x.ndim != 5 or x.shape[1] < 2 or x.shape[2] != self.in_channels:
            raise ValueError(f"x must be [B,T>=2,{self.in_channels},H,W]")
        b, t, c, h, w = x.shape
        if h % 16 or w % 16:
            raise ValueError("SalsaNext requires H and W divisible by 16")
        if valid.shape == (b, t, 1, h, w):
            valid = valid[:, :, 0]
        if valid.shape != (b, t, h, w) or dt.shape != (b, t):
            raise ValueError("valid must be [B,T,H,W] and dt [B,T]")
        if not torch.isfinite(dt).all() or not (dt[:, :-1] < 0).all() or not (dt[:, -1] == 0).all():
            raise ValueError("Past dt must be negative seconds and current dt zero")
        encoded, skips = self.encoder(x.reshape(b * t, c, h, w))
        _, f, fh, fw = encoded.shape
        encoded = encoded.reshape(b, t, f, fh, fw)
        semantic = None
        if self.movable_branch == 'current_bottleneck' and (return_aux or self.semantic_fusion == 'bottleneck_gate'):
            # Current scan BEFORE temporal attention; same encoder forward.
            semantic = self.semantic_adapter(encoded[:, -1])
        scaled_valid = F.adaptive_max_pool2d(valid.reshape(b * t, 1, h, w).float(), (fh, fw)) > 0.5
        scaled_valid = scaled_valid.reshape(b, t, 1, fh, fw)
        result = self.temporal(encoded[:, -1], encoded[:, :-1], scaled_valid[:, -1],
                               scaled_valid[:, :-1], dt[:, :-1], return_diagnostics)
        if return_diagnostics:
            fused, diagnostics = result
        else:
            fused = result
        if self.semantic_fusion == 'bottleneck_gate':
            modulation = torch.tanh(self.semantic_gate(semantic))
            fused = fused + 0.1 * fused * modulation * scaled_valid[:, -1].to(fused.dtype)
        current_skips = tuple(s.reshape(b, t, *s.shape[1:])[:, -1] for s in skips)
        if return_aux and self.movable_branch == 'current_bottleneck':
            logits = self.decoder(fused, current_skips)
            output = (logits, self.movable_head(semantic, current_skips[2], (h, w)))
        elif return_aux:
            logits, features = self.decoder(fused, current_skips, return_features=True)
            head = self.movable_head if self.auxiliary_movable else self.residual_head
            output = (logits, head(features))
        else:
            logits = self.decoder(fused, current_skips)
            output = logits
        if logits.shape != (b, self.num_classes, h, w):
            raise RuntimeError(f"Unexpected decoder output {tuple(logits.shape)}")
        return (output, diagnostics) if return_diagnostics else output
