# Supervised temporal SalsaNext MOS

Run these commands from `/home/devuser/workspace`, the local Git root containing
`src/` and `tools/`.

## Architecture and data

This is a range-view MOS adaptation inspired by temporal MAE, not a reproduction
of its pillar masking or reconstruction objective. The default starts from
scratch. A single shared SalsaNext encoder processes `[B,T,4,H,W]` scans
(`x,y,z,range`), with the current scan at index `-1`. At the bottleneck,
`[B,T,256,H/16,W/16]`, current features provide queries and only past frames
provide local keys and values. The default 3×3 neighborhood wraps in azimuth
and pads elevation with masked invalid positions. A learned no-match key/value
ensures finite attention when history is empty. A shared comparison projection
forms current minus matched-history features; context, difference, no-match
weight, best historical match, expected `dy/dx/dt`, and entropy feed a change
MLP. A channelwise residual scale starts at `1e-3`. Only the bottleneck is
fused; the SalsaNext decoder receives current-frame skips and returns raw
`[B,2,H,W]` logits. No-match weight is an attention diagnostic, not a calibrated
motion probability.

`TemporalMOSE2EDataset` uses the existing SemanticKITTI sequence/pose and
range-view projection functions. It appends a constant occupancy channel
before projection so validity follows the exact collision resolution. Past
scans are transformed with `inv(current_pose) @ past_pose`; the pose loader
converts camera odometry into Velodyne coordinates using calibration. The
dataset returns `x`, explicit `valid`, `dt`, current `y`, and `meta`. Invalid
current projection cells receive label `-1`. The default offset `[3]` maps to
approximately `-0.3 s` using nominal 10 Hz; these are estimates, not measured
timestamps. For T=3, set `previous_offsets: [4, 2]` in a copy of the YAML.

## Commands

```bash
python tools/test_temporal_mos_e2e.py
python tools/train_temporal_mos_e2e.py --device cpu --dry-run
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e.yaml \
  --device cuda --epochs 1 --batch_size 1 --num_workers 0 --max_batches 1
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e.yaml --device cuda
python tools/evaluate_temporal_mos_e2e.py \
  --checkpoint PATH_TO_RUN/checkpoints/best_moving_iou.pt --split val --device cuda
python tools/evaluate_temporal_mos_e2e.py \
  --checkpoint PATH_TO_RUN/checkpoints/best_moving_iou.pt \
  --split val --history-mode zero --device cuda
python tools/evaluate_temporal_mos_e2e.py \
  --checkpoint PATH_TO_RUN/checkpoints/best_moving_iou.pt \
  --split val --history-mode copy_current --device cuda
```

Use `--help` for all CLI options. The normal training command targets 50
epochs. `--dry-run` is synthetic and creates no log directory. A real run
creates `mos_logs/MOS_temporal_e2e_scratch_YYYYMMDD_HHMMSS/` with a YAML copy,
`csv/epochs.csv`, `checkpoints/last.pt`,
`checkpoints/best_moving_iou.pt`, and optional `tensorboard/` events. The best
checkpoint is selected only by validation Moving IoU. The evaluator strictly
loads the model state and writes `eval/{split}_{history_mode}.csv`.

Train/val/test are sequences `00–06,09,10` / `08` / `07`, with no overlap.
Weighted cross entropy uses current labels only and ignores `-1`. All encoder,
attention, change, decoder, and head parameters are in one AdamW group. The
YAML uses seed 42, LR `5e-4`, weight decay `1e-4`, gradient clip `5`, weights
`[1,10]`, cosine decay and 50 epochs. AMP defaults to bfloat16 on the
RTX 4070 Ti because its wider exponent range avoids float16 activation
overflow; `amp_dtype: float16` and `amp: false` remain available for comparison
or older GPUs. Gradient accumulation is configurable. The training loop checks
the gradient norm before each unscaled optimizer step and reports sample IDs
if a loss becomes non-finite. CSV records measured timing and peak allocated VRAM
for each phase; the timing includes batch work and should be treated as a
rough efficiency indicator.

For fair comparisons, run single-frame SalsaNext scratch, the existing
two-frame `salsanext_temporal_mos` scratch control, and this model with the
same splits, offset, input channels, valid target frames, seed, batch/optimizer
schedule, and full training budget. The default batch size is 4, matching the
existing temporal scratch control; use `--batch_size 1` for a smoke run. The
single-frame default YAML currently
has `min_range: 0.0`, while this path and the temporal scratch control use
`0.1`; align that setting before claiming a controlled comparison. The zero
and copy-current history modes are diagnostics; identical or different metrics
on a small sample do not establish temporal effectiveness. Parameter count,
peak VRAM, and timing can differ between architectures and must be reported.
