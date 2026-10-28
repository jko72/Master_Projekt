# Temporal E2E MOS: isolated CE / CE + Lovasz ablation

Only `tools/train_temporal_mos_e2e.py` uses the new optional loss. Encoder,
attention, decoder, labels, projection, optimizer, scheduler, splits and best
validation Moving-IoU checkpoint selection are unchanged. Existing local edits
to the trainer and configs were preserved. No full training job was submitted.

## Reference inspection and licensing

Inspected on 2026-10-26:

- [LMNet trainer](https://github.com/PRBonn/LiDAR-MOS/blob/main/mos_SalsaNext/train/tasks/semantic/modules/trainer.py),
  file blob `699204b4f330aad7c5b9d02b7ef3f67faca60c20`:
  weighted NLL on log probabilities plus Lovasz on probabilities.
- [LMNet Lovasz](https://github.com/PRBonn/LiDAR-MOS/blob/main/mos_SalsaNext/train/tasks/semantic/modules/Lovasz_Softmax.py),
  blob `2b4154111523c4ef22e44f5e2f5360c7dd9e4506`.
- [MF-MOS trainer](https://github.com/SCNU-RISLAB/MF-MOS/blob/main/modules/trainer.py),
  blob `3b3c79a3fc0f9dd872f69d4fe77b4a33b1b72c52`:
  moving CE + moving Lovasz + movable CE + movable Lovasz.
- [MF-MOS Lovasz](https://github.com/SCNU-RISLAB/MF-MOS/blob/main/modules/loss/Lovasz_Softmax.py),
  blob `42a6d4ee3fe0bc6132f4ea09d17f26bb7c47bb0b`.
- [MF-MOS label config](https://github.com/SCNU-RISLAB/MF-MOS/blob/main/config/labels/semantic-kitti-mos.raw.yaml),
  blob `4758997fe76a7ec1074ff0e683191652bef3a9c3`: separate moving/movable
  learning maps. Their encoding is ignore=0, static=1, moving=2; ours remains
  ignore=-1, static=0, moving=1. Their frequency-derived weights are not adopted.

The reference Lovasz files carry MIT notices for Maxim Berman (2018) and
Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy (2020). The notice and
MIT terms are retained in `LICENSES/LovaszSoftmax-MIT.txt`. No trainer code was
copied. The original LMNet loss is stored unchanged as a test-only numerical
oracle in `tests/reference/lmnet_lovasz.py`, with its original notice. Production
does not import it. Tests compare values and gradients for ordinary small cases;
its known single-valid-pixel indexing and empty-loss defects are tested separately
against exact expected behavior in the new implementation.

## Loss and numerical policy

For valid pixels V, weights w_0=1 and w_1=10, original logits z and softmax p:

    CE = sum_{i in V} w_{y_i} [-log softmax(z_i)_{y_i}] / sum_{i in V} w_{y_i}
    total = CE + lambda_lovasz * Lovasz,    lambda_lovasz = 1 by default.

CE uses the original native PyTorch CrossEntropyLoss and its original autocast
policy. Lovasz explicitly disables autocast and computes
`torch.softmax(logits.float(), dim=1)`. For each selected class c, define
f_i = 1[y_i=c], e_i=|f_i-p_ic|. Sort errors descending, applying the same
permutation to f. At sorted position k, with G=sum(f):

    intersection_k = G - sum_{j<=k} f_j
    union_k        = G + sum_{j<=k} (1-f_j)
    J_k            = 1 - intersection_k / union_k
    delta_J        = [J_1, J_2-J_1, ..., J_P-J_{P-1}]
    Lovasz_c       = dot(sorted_errors, delta_J).

Average equally across selected classes. `present` selects classes represented
in local valid ground truth, `all` selects both. There is no moving-only term
and no CE weight in Lovasz. `per_image: false` flattens the local microbatch;
`true` averages image losses (including zero for fully ignored images).
Ignored pixels are filtered by a Boolean mask before sorting, preserving [P,2]
even for P=1. An all-ignore loss call returns a differentiable scalar zero;
the existing training loop still rejects an all-ignore batch with
`ValueError("Batch contains no supervised current pixels")` in either mode.
Non-finite input logits are still a data/model error, not silently repaired.

The scalar criterion forward works with existing dry-run and evaluation calls;
`components(logits, target)` exposes `loss_ce`, `loss_lovasz`, `loss_total`.
No trainable parameters or pixel subsampling are introduced.

## Configuration and backward compatibility

Missing `mos_train_params.loss` resolves to `type: ce`. The CE-only factory
returns the exact original `torch.nn.CrossEntropyLoss`, without computing softmax
or Lovasz. `read_config()` resolves all loss defaults so `config.yaml` and the
existing checkpoint `config` payload contain the complete settings. Unknown keys,
types, negative/non-finite weights and non-Boolean per-image values are rejected.
`lovasz_weight: 0` gives identical optimization loss and logits gradients to CE,
while still computing/logging the unweighted Lovasz for diagnostics.

The new complete config is
`src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml`.
Its reference is `mos_semanticKitti_temporal_e2e_intensity_off8_p2_fullres_2gpu.yaml`:
100 epochs, offset 8, XYZRI, 64x2048, dropout .1, weights 1/10, LR .0005,
AdamW/cosine, batch 8 per GPU, two GPUs, accumulation 1, **AMP false**,
no pretraining. Only loss settings and run name change. This equality is tested.
The other fullres YAML is batch 4 and is not the requested global-batch-16 run.

## DDP, accumulation, validation and logging

For rank r, microstep t, let W_rt be the sum of valid CE class weights and R
the number of ranks. The backward expression before accumulation division is:

    CE_rt * R * W_rt / sum_r W_rt + lambda * Lovasz_rt.

DDP's gradient average therefore gives the global weighted CE gradient plus
the mean of the local Lovasz gradients. CE's weight correction is never applied
to Lovasz. No all-gather of full-resolution probabilities is needed. Local
Lovasz averaging is NOT equivalent to Lovasz on the concatenated global batch:
ordering and present-class selection happen separately on each rank/microbatch.
Comparisons should retain world size, per-rank batch and accumulation settings.

Both terms are divided by the same actual accumulation group size, including
the last short group. `no_sync()`, unscale, clipping and GradScaler are retained.
As before, accumulation averages global CE microbatch means; with unequal CE
denominators this differs from CE over a single concatenated accumulation group.
Lovasz similarly averages local microbatch objectives and does not increase its
weight when accumulation increases. It is not sorted over the accumulated batch.

Epoch reporting is explicitly distinct from the per-update objective:

- `loss_ce`: sum of CE numerators / sum of CE weight denominators across all
  processed batches and ranks, exactly the previous epoch CE statistic.
- `loss_lovasz`: arithmetic mean of local batch Lovasz values across all
  processed rank-batches, without pixel/CE/sample weighting. A smaller final
  batch has equal weight. With per-image mode, image means are still averaged
  per local batch.
- `loss_total`: `loss_ce + lambda * loss_lovasz`; existing `loss` aliases total.
  In CE-only mode Lovasz is zero and total equals the original CE statistic.

The existing dictionary-driven CSV/TensorBoard writers produce the six added
`train_loss_*` / `val_loss_*` fields. Existing MOS count reductions, timings,
VRAM maxima, learning rate and metrics remain. Validation has no per-batch
collectives, so unequal and empty shards remain supported. Loss sums/counts are
reduced only at phase end. Global metrics remain computed from summed counts.

## Validation performed (2026-10-26)

Isolated CPU environment: Python 3.12, PyTorch 2.14.1+cpu, two CPU threads.
Dependencies installed only in `/tmp/mos-lovasz-venv`; HPC environment unchanged.

- `python -m unittest discover -s tests -v`: 26 passed, CUDA test skipped;
  DDP initially could not initialize loopback inside the sandbox.
- DDP rerun outside sandbox with `GLOO_SOCKET_IFNAME=lo`: passed (two processes).
  Includes CE-only global-batch regression; combined objective parameter-update
  comparison; unequal class ratios; accumulation with a short final group;
  uneven/empty validation; CSV and checkpoint reload; two epochs; real temporal
  model two DDP backwards; injected overflow recovery.
- `python tools/test_temporal_mos_e2e.py`: all 9 existing tests passed.
- New combined-config `tools/train_temporal_mos_e2e.py --dry-run --device cpu`:
  passed in one attempt, T=2, logits [1,2,32,64], loss 1.38183.
- Full real-model CE+Lovasz forwards/backwards at B=1:

| Resolution | Forward/backward seconds | CE | Lovasz | Total | Gradient norm before clipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x512 | 1.647 | .86878 | .57406 | 1.44284 | 7.912 |
| 64x2048 | 7.312 | .71008 | .57256 | 1.28265 | 3.351 |

Both resolutions had finite gradients on every trainable parameter and nonzero
gradients in encoder, temporal attention and decoder. All reported losses and
gradients were finite. The fullres model smoke uses B=1, not the HPC B=8/GPU.
CPU BF16 autocast verifies that Lovasz stays FP32. CUDA FP16 AMP test is provided
but skipped because no CUDA driver/device is available; CPU GradScaler overflow
recovery is tested, and does not establish CUDA FP16 hardware validation.
`compileall` and `git diff --check` passed.

## Performance and memory

Synthetic loss forward + backward to logits, B=8, 80% valid, approximately 2%
moving pixels, FP32, two CPU threads, three warmups and ten measured iterations:

| Resolution | CE median ms | CE+Lovasz median ms | Additional ms |
| --- | ---: | ---: | ---: |
| 64x512 | 3.66 | 50.92 | 47.26 |
| 64x2048 | 15.85 | 210.42 | 194.57 |

CPU timings are not estimates of A100 timings or total-epoch overhead.
GPU allocation could not be measured here. Complexity: two class sorts,
O(P log P) work and O(N+P) tensor/workspace memory for N=BHW and P valid pixels.
At B=8, 64x2048, FP32 probabilities alone occupy 8 MiB; one int64 permutation
at 80% valid occupies 6.4 MiB, and both classes' sort indices may be retained
for backward. Errors, gradients, masks, sort workspaces and FP32 logits cast
add further memory: expect additional tens of MiB, potentially over 100 MiB
depending on CUDA sorting/autograd/allocator. This is an allocation-size estimate,
not a measured GPU peak or guaranteed upper bound.

Reproducible measurement on a GPU node, in the existing training container:

```bash
python3 tools/benchmark_temporal_mos_loss.py --device cuda --batch-size 8 --iterations 20
python3 tools/benchmark_temporal_mos_loss.py --device cuda --batch-size 1 --model-smoke
# Optional FP16 validation, separate from the AMP=false baseline:
python3 tools/benchmark_temporal_mos_loss.py --device cuda --batch-size 1 --model-smoke --amp
```

The benchmark emits JSON including CUDA incremental allocated peak for each
criterion and peak difference, with synchronization and warmups. Real model
smoke reports total peak allocation and gradient health. Cached/reserved memory
and total process VRAM are different quantities. Run alone for meaningful timings.

## HPC invocation (not executed)

From the repository directory, use the unchanged two-GPU wrapper:

```bash
bash submit_mos_p2_2gpu.sh src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml
```

First validate CUDA without starting 100 epochs:

```bash
bash submit_mos_p2_2gpu.sh src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml --dry-run
```

The wrapper keeps the reference p2/two-GPU setup, container bindings and log
layout. Direct entry inside that container is:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 tools/train_temporal_mos_e2e.py --device cuda --cfg_path src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml
```

Outstanding: CUDA FP16/NCCL validation and measured GPU loss-time/VRAM on the HPC
node; no scientific improvement claim until the controlled full run completes.
