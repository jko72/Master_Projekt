from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn

try:
    from .salsanext_parts import SalsaNextDecoder, SalsaNextEncoder, adapt_first_conv_in_channels
except ImportError:  # direct file execution fallback
    from salsanext_parts import SalsaNextDecoder, SalsaNextEncoder, adapt_first_conv_in_channels


def _clean_state_dict_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> "OrderedDict[str, torch.Tensor]":
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        nk = str(k)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
                    changed = True
                    break
        clean_state[nk] = v
    return clean_state


def _strip_decoder_head(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple["OrderedDict[str, torch.Tensor]", list[str]]:
    clean_state = OrderedDict()
    skipped = []
    for k, v in state_dict.items():
        nk = str(k)
        if nk.startswith("logits."):
            skipped.append(nk)
            continue
        clean_state[nk] = v
    return clean_state, skipped


class SalsaNextMOS(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.encoder = SalsaNextEncoder(in_channels=self.in_channels, dropout=self.dropout)
        self.decoder = SalsaNextDecoder(num_classes=self.num_classes, dropout=self.dropout)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, -1]
        elif x.ndim != 4:
            raise ValueError(
                f"SalsaNextMOS expects input with 4D [B,C,H,W] or 5D [B,T,C,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SalsaNextMOS expected {self.in_channels} input channels, but got {x.shape[1]} "
                f"for tensor shape={tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        bottleneck, skips = self.encoder(x)
        logits = self.decoder(bottleneck, skips)
        return logits

    def load_pretrained_encoder(
        self,
        encoder_state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        adapt_input_channels: bool = True,
        init_new_channels: str = "mean",
    ):
        clean_state = _clean_state_dict_prefixes(
            encoder_state_dict,
            prefixes=("model.", "module.", "encoder."),
        )

        if adapt_input_channels:
            clean_state = adapt_first_conv_in_channels(
                clean_state,
                target_in_channels=self.in_channels,
                conv_key_hint="down_cntx.conv1.weight",
                mode=init_new_channels,
            )
            missing, unexpected = self.encoder.load_state_dict(clean_state, strict=False)
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"Strict encoder load failed after input-channel adaptation. "
                    f"Missing keys: {missing} | Unexpected keys: {unexpected}"
                )
        else:
            missing, unexpected = self.encoder.load_state_dict(clean_state, strict=strict)

        print(
            f"[PRETRAIN] Encoder weights loaded into SalsaNextMOS. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[PRETRAIN] Missing keys (sample): {missing[:8]}")
        if unexpected:
            print(f"[PRETRAIN] Unexpected keys (sample): {unexpected[:8]}")
        return missing, unexpected

    def load_pretrained_backbone(
        self,
        encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        decoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        strict_encoder: bool = True,
        strict_decoder: bool = False,
        adapt_input_channels: bool = True,
        init_new_channels: str = "zero",
        skip_decoder_head: bool = True,
    ):
        result = {
            "encoder_missing": [],
            "encoder_unexpected": [],
            "decoder_missing": [],
            "decoder_unexpected": [],
        }

        if encoder_state_dict is not None:
            enc_missing, enc_unexpected = self.load_pretrained_encoder(
                encoder_state_dict=encoder_state_dict,
                strict=strict_encoder,
                adapt_input_channels=adapt_input_channels,
                init_new_channels=init_new_channels,
            )
            result["encoder_missing"] = list(enc_missing)
            result["encoder_unexpected"] = list(enc_unexpected)

        if decoder_state_dict is not None:
            clean_decoder_state = _clean_state_dict_prefixes(
                decoder_state_dict,
                prefixes=("model.", "module.", "decoder."),
            )
            skipped_head_keys = []
            if skip_decoder_head:
                clean_decoder_state, skipped_head_keys = _strip_decoder_head(clean_decoder_state)

            missing, unexpected = self.decoder.load_state_dict(clean_decoder_state, strict=False)
            result["decoder_missing"] = list(missing)
            result["decoder_unexpected"] = list(unexpected)

            print(
                f"[PRETRAIN] Decoder weights loaded into SalsaNextMOS. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            print(f"[PRETRAIN] Skipped decoder head keys: {skipped_head_keys}")
            if missing:
                print(f"[PRETRAIN] Decoder missing keys (sample): {missing[:8]}")
            if unexpected:
                print(f"[PRETRAIN] Decoder unexpected keys (sample): {unexpected[:8]}")

            if strict_decoder:
                allowed_missing = {"logits.weight", "logits.bias"} if skip_decoder_head else set()
                extra_missing = [k for k in missing if k not in allowed_missing]
                if extra_missing or unexpected:
                    raise RuntimeError(
                        f"Strict decoder load failed. Missing keys: {missing} | Unexpected keys: {unexpected}"
                    )

        return result


if __name__ == "__main__":
    model = SalsaNextMOS(in_channels=2, num_classes=2)
    x = torch.randn(2, 2, 64, 512)
    y = model(x)
    print(y.shape)
    assert y.shape == (2, 2, 64, 512)
