# Train-only Temporal-E2E-MOS augmentation

The dataset API stays `x, valid, dt, y, meta`. No temporal offset sampling is
implemented. History indices, poses, `dt`, model, losses, KNN, threshold selection,
and checkpoint selection are unchanged.

Add this optional top-level section to any Temporal E2E training YAML:

```yaml
mos_augmentation:
  enabled: false
  azimuth_rotation:
    enabled: false
    probability: 0.5
  historical_point_dropout:
    enabled: false
    probability: 0.5
    max_dropout_ratio: 0.10
  intensity_jitter:
    enabled: false
    probability: 0.5
    std: 0.02
```

These are the defaults for missing parameters. Set the global `enabled: true`
and each desired method's `enabled: true`; leave the other methods false for
isolated experiments. All three true enables the combined experiment.
Probabilities are independent: rotation is sampled once per sample, dropout
once per historical scan, and jitter once per frame. Probability zero skips the
method. All switches must be booleans; probability and maximum dropout are
finite numbers in [0,1]; jitter std is finite and nonnegative. Unknown keys are
rejected to catch misspellings. Active jitter requires `range_xyz_intensity`.
Global false overrides individual switches.

Augmentation requires both `split == "train"` and the global and individual
switches. Validation and test consume no augmentation RNG draws and match the
original path. The checkpoint evaluator additionally disables augmentation even
when evaluating the train split. Official benchmark, KNN evaluation, and
threshold sweeps use their existing independent input builders, which do not
invoke these augmentation helpers. Labels on disk are never edited.

## Geometry and ordering

1. Existing ego-motion compensation transforms each historical XYZI scan into
   current Velodyne coordinates. If dropout fires, a fresh rate is sampled
   uniformly between zero and `max_dropout_ratio` for that scan. Independent
   Bernoulli removal uses that rate for every raw point. The realized fraction
   fluctuates statistically (the configured maximum bounds the sampled removal
   probability, not the realized fraction). Only remaining points enter the
   existing projection, including near-point collision resolution and the extra
   occupancy channel. Current points and labels are untouched by dropout.
2. Jitter adds independent Gaussian noise to valid raw remission pixels on
   every selected frame. SemanticKITTI float remission is nominally [0,1];
   changed values are clipped to [0,1] before normalization. Invalid cells and
   XYZR stay unchanged. Noise arithmetic uses CPU float64 to avoid float32
   overflow for unusually large finite std values, then casts back to float32.
3. A rotation draws an integer shift uniformly from `[0, W-1]` and rolls all
   frame features, masks, and the current label image identically with wrap.
   Valid XY values rotate by `angle = -2*pi*shift/W`, while Z, range, and
   remission are simply rolled. No vertical transformation is used.
4. Existing feature normalization follows all augmentation. Invalid cells remain
   masked, and current labels are masked with the transformed current occupancy.

The sign follows the actual `utils_torch.spherical_projection`: azimuth is
`atan2(y,x)` and columns reverse the ascending angle-bin indices. Its bins use
`linspace(-pi, pi, W)`, spacing `2*pi/(W-1)`, with endpoint asymmetry and clipping.
A cyclic W-column image therefore cannot exactly represent raw rigid rotation
and re-projection everywhere. We explicitly use a periodic image convention
(W columns = 2*pi). This preserves cylindrical wrap, a common rigid XY transform,
range norms, and exact pixel/label/mask alignment; it does not guarantee identical
bins to re-projecting rotated raw points, especially at the seam. The original
projection is unchanged.

## Seeds and saved configuration

All draws use the CPU PyTorch RNG; no CUDA operations are added to the dataset.
The existing training entry seeds torch with `seed + rank`. DataLoader already
seeds each worker's torch RNG from its dedicated `Generator(seed + rank)` and
worker ID; a custom worker initializer is unnecessary. Recreated workers get
new deterministic seeds as the loader generator advances each epoch. The
DistributedSampler retains its existing `set_epoch` behavior. Distinct samples,
workers, and ranks can draw different augmentation. With zero workers the rank's
main torch RNG is used. Disabled methods consume no random numbers, preserving
the previous training RNG stream. Reproducibility assumes the same seed, rank
count, workers, sample order, library versions, and deterministic-kernel settings;
bitwise resume from an arbitrary checkpoint is not newly guaranteed.

`read_config` resolves the entire augmentation section (including defaults), so
existing saved `config.yaml` and checkpoint config logging record all parameters.
The source YAML is copied as before.

## Reference configuration and commands

`src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz_w290_aug.yaml`
is copied from the local `...off8_fullres_lovasz_w290.yaml`: 64 x 2048, offset [8],
CE+Lovasz, class weights [1,290]. Only the augmentation section and run name differ;
all other hyperparameters remain identical.

From the repository root, in the existing training environment/container:

```bash
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz_w290_aug.yaml \
  --device cuda
```

The YAML retains the reference container paths `/hpc/SemanticKitti/...` and
`/hpc/logs/mos`; the host environment must provide those paths as before.

Tests, including real single-step CPU FP32 training at 64 x 512 and 64 x 2048:

```bash
python -m unittest discover -s tests -p 'test_temporal_mos_augmentation.py' -v
python tools/test_temporal_mos_e2e.py
python -m unittest discover -s tests -p 'test_mos_*.py'
python -m unittest discover -s tests -p 'test_temporal_mos_distributed.py'
```

Training smoke tests use synthetic scans through the real dataset, projection,
model, CE+Lovasz, backward, and AdamW update (batch 1, two frames). Multiworker
and Gloo/DDP checks require permission to create local sockets. No full HPC
training or GPU/DDP performance claim is made by these tests.

## Local verification (2026-10-28)

- New augmentation suite: 13 tests passed, including two Gloo ranks with two
  DataLoader workers each, reproducibility, evaluation exclusion, saved YAML/
  checkpoint settings, and the real model training smoke at both resolutions.
- Existing MOS suites: 47 tests, 45 passed, 2 CUDA-only tests skipped.
- Existing temporal E2E checks: 9 passed.
- Existing temporal distributed suite: 1 passed (two CPU/Gloo processes).
- CPU FP32 augmented training losses: 1.36530 (64 x 512), 1.22782 (64 x 2048),
  both with finite backward gradients and an actual AdamW update.
- `git diff --check` passed. GPU execution was unavailable.

The new experiment YAML follows the existing local `.git/info/exclude` rule
for HPC-only YAML files: it exists locally and is intentionally not staged.
