#!/usr/bin/env python3
"""Synthetic CE/Lovasz timing and optional real temporal forward/backward.

No dataset, logs, model checkpoints or training jobs are created. Timings cover
loss forward and backward to logits; CUDA memory is incremental allocated peak
above input tensors, not total training VRAM. No subsampling is used.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from helper.mos_lovasz_loss import MOSCrossEntropyLovasz


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_loss(device, batch_size, width, iterations, amp):
    logits = torch.randn(batch_size, 2, 64, width, device=device,
                         dtype=torch.float16 if amp else torch.float32, requires_grad=True)
    target = torch.zeros(batch_size, 64, width, dtype=torch.long, device=device)
    random = torch.rand_like(target, dtype=torch.float32)
    target[random < 0.02] = 1
    target[random > 0.8] = -1
    del random
    weights = torch.tensor([1., 10.], device=device)
    criteria = dict(ce=torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1),
                    ce_lovasz=MOSCrossEntropyLovasz(weights))
    measurements = {}
    for name, criterion in criteria.items():
        durations, peaks = [], []
        for index in range(iterations + 3):
            logits.grad = None
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(logits, target)
            loss.backward()
            synchronize(device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else None
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
            if index >= 3:
                durations.append(elapsed * 1000)
                if device.type == "cuda":
                    peaks.append(peak)
            del loss
        measurements[name] = dict(median_forward_backward_ms=statistics.median(durations),
                                  incremental_peak_allocated_bytes=max(peaks) if peaks else None)
    measurements["extra_ms"] = (measurements["ce_lovasz"]["median_forward_backward_ms"]
                                - measurements["ce"]["median_forward_backward_ms"])
    if device.type == "cuda":
        measurements["extra_peak_bytes"] = (measurements["ce_lovasz"]["incremental_peak_allocated_bytes"]
                                            - measurements["ce"]["incremental_peak_allocated_bytes"])
    return measurements


def model_smoke(cfg_path, device, width, amp):
    from train_temporal_mos_e2e import read_config, make_criterion, gradient_health
    from mos_models import build_mos_model
    _, cfg = read_config(cfg_path)
    cfg["mos_train_params"]["loss"] = dict(type="ce_lovasz")
    model = build_mos_model(cfg).to(device).train()
    offsets = sorted(cfg["mos_data_params"]["previous_offsets"], reverse=True)
    x = torch.randn(1, len(offsets) + 1, cfg["mos_model_params"]["in_channels"], 64, width, device=device)
    valid = torch.ones(1, len(offsets) + 1, 64, width, dtype=torch.bool, device=device)
    dt = torch.tensor([[-v / cfg["mos_data_params"].get("scan_rate_hz", 10.) for v in offsets] + [0.]], device=device)
    target = torch.randint(0, 2, (1, 64, width), device=device)
    target[:, :4] = -1
    criterion = make_criterion(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        parts = criterion.components(model(x, valid, dt), target)
    # Lower initial scale for this single-step benchmark; the production dry-run
    # separately tests automatic overflow recovery with the default scale.
    if amp:
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        scaler.scale(parts["loss_total"]).backward()
        scaler.unscale_(optimizer)
    else:
        parts["loss_total"].backward()
    synchronize(device)
    health = gradient_health(model)
    assert not any(health.values()), health
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    return dict(seconds=time.perf_counter() - start, gradient_norm_before_clipping=float(norm),
                gradient_health=health, **{k: float(v.detach()) for k, v in parts.items()},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model-smoke", action="store_true", help="Real model, B=1, each requested resolution")
    parser.add_argument("--cfg-path", default=str(ROOT / "src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        parser.error("--amp requires CUDA")
    if min(args.batch_size, args.iterations, args.threads, *args.widths) <= 0:
        parser.error("batch-size, iterations, threads and widths must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    print(json.dumps(dict(torch=torch.__version__, device=str(device), threads=args.threads,
                          gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          batch_size=args.batch_size, amp=args.amp)), flush=True)
    for width in args.widths:
        print(json.dumps(dict(width=width, loss=benchmark_loss(device, args.batch_size, width,
                                                            args.iterations, args.amp))), flush=True)
        gc.collect()
        if args.model_smoke:
            print(json.dumps(dict(width=width, model=model_smoke(args.cfg_path, device, width, args.amp))), flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
