"""FP32 Lovasz-Softmax for binary MOS, with optional weighted CE.

Algorithm adapted from Maxim Berman's LovaszSoftmax and the LMNet/MF-MOS
implementations. See TEMPORAL_MOS_LOVASZ_ABLATION.md and LICENSES/LovaszSoftmax-MIT.txt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


def loss_settings(value=None):
    """Resolve legacy CE defaults and reject misspelled/invalid loss settings."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("mos_train_params.loss must be a mapping")
    settings = dict(type="ce", lovasz_weight=1.0, lovasz_classes="present", lovasz_per_image=False)
    unknown = set(value) - set(settings)
    if unknown:
        raise ValueError(f"Unknown mos_train_params.loss settings: {sorted(unknown)}")
    settings.update(value)
    if settings["type"] not in ("ce", "ce_lovasz"):
        raise ValueError("mos_train_params.loss.type must be ce or ce_lovasz")
    weight = settings["lovasz_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("mos_train_params.loss.lovasz_weight must be finite and nonnegative")
    if settings["lovasz_classes"] not in ("present", "all"):
        raise ValueError("mos_train_params.loss.lovasz_classes must be present or all")
    if not isinstance(settings["lovasz_per_image"], bool):
        raise ValueError("mos_train_params.loss.lovasz_per_image must be a YAML boolean")
    settings["lovasz_weight"] = float(weight)
    return settings


def lovasz_grad(foreground_sorted):
    """Discrete increments of Jaccard loss along decreasing prediction errors."""
    positives = foreground_sorted.sum()
    intersection = positives - foreground_sorted.cumsum(0)
    union = positives + (1.0 - foreground_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    return torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))


def _flat_loss(probabilities, target, classes, ignore_index):
    labels = target.reshape(-1)
    valid = labels != ignore_index
    # Boolean indexing retains [P,2], also when P=1. Filter before sorting.
    probabilities = probabilities.movedim(1, -1).reshape(-1, 2)[valid]
    labels = labels[valid]
    if labels.numel() == 0:
        return probabilities.sum() * 0.0
    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("MOS target must contain only 0 (static), 1 (moving), or ignore_index")
    losses = []
    for class_index in range(2):
        foreground = (labels == class_index).float()
        if classes == "present" and not foreground.any():
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        errors_sorted, permutation = errors.sort(descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground[permutation])))
    return torch.stack(losses).mean()


class MOSLovaszSoftmax(nn.Module):
    """Lovasz on softmax(logits.float()), without CE class weights.

    per_image=False sorts all valid pixels of the local batch together.
    per_image=True averages image losses, including zero for ignored images.
    """

    def __init__(self, classes="present", per_image=False, ignore_index=-1):
        super().__init__()
        loss_settings(dict(lovasz_classes=classes, lovasz_per_image=per_image))
        self.classes, self.per_image, self.ignore_index = classes, per_image, ignore_index

    def forward(self, logits, target):
        if logits.ndim != 4 or logits.shape[1] != 2 or target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError("Expected MOS logits [B,2,H,W] and target [B,H,W]")
        if target.dtype != torch.long:
            raise ValueError("MOS target must have dtype torch.long")
        # Explicitly disable autocast: dot/sort/softmax must stay FP32, including
        # callers running the whole criterion under FP16 or BF16 autocast.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = torch.softmax(logits.float(), dim=1)
            if self.per_image and logits.shape[0] > 0:
                return torch.stack([
                    _flat_loss(p.unsqueeze(0), t.unsqueeze(0), self.classes, self.ignore_index)
                    for p, t in zip(probabilities, target)
                ]).mean()
            return _flat_loss(probabilities, target, self.classes, self.ignore_index)


class MOSCrossEntropyLovasz(nn.CrossEntropyLoss):
    """Scalar forward for compatibility; components() exposes all three losses.

    Inherits CE's weight/ignore_index interface for the existing trainer. CE
    receives original logits under the caller's existing autocast policy.
    """

    def __init__(self, weight, lovasz_weight=1.0, classes="present", per_image=False):
        super().__init__(weight=weight, ignore_index=-1)
        loss_settings(dict(lovasz_weight=lovasz_weight))
        self.lovasz_weight = float(lovasz_weight)
        self.lovasz = MOSLovaszSoftmax(classes, per_image)

    def components(self, logits, target):
        # Empty slicing avoids summing extreme logits just to create a zero.
        ce = super().forward(logits, target) if (target != -1).any() else logits.float().reshape(-1)[:0].sum()
        lovasz = self.lovasz(logits, target)
        total = ce + self.lovasz_weight * lovasz
        return dict(loss_ce=ce, loss_lovasz=lovasz, loss_total=total)

    def forward(self, logits, target):
        return self.components(logits, target)["loss_total"]


def loss_components(criterion, logits, target):
    """Keep native CrossEntropyLoss unchanged for legacy CE-only experiments."""
    if isinstance(criterion, MOSCrossEntropyLovasz):
        return criterion.components(logits, target)
    ce = criterion(logits, target)
    return dict(loss_ce=ce, loss_lovasz=ce.detach().new_zeros(()), loss_total=ce)
