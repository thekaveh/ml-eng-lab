# 8.8 Quantization — MNIST FFN (PTQ + QAT via torchao)

A comprehensive walk-through of `notebooks/quantization-mnist-ffnn-pytorch/` — the canonical
in-repo demo of post-training quantization (`nnx.quantize_int8`) and quantization-aware training
(`nnx.qat_train_step_factory` + `nnx.QATLifecycleCallback`), both backed by `torchao`. This page is
the deep-dive companion to the task notebook: it states the problem, builds the quantization math,
dissects the PTQ vs QAT contracts, reads the code top to bottom, reports the measured
accuracy / size / latency comparison, and catalogues the pitfalls and extensions.

The notebook is **manual-only** — it is *not* in the Tier-A/B/C papermill targets and is not
re-executed in CI. The reason is an upstream dependency pin: `torchao`'s `Int8WeightOnlyConfig`
references `torch.int1`, which only exists in `torch >= 2.5`, while ml-eng-lab pins
`torch==2.4.1` for genai-vanilla image parity. No `torchao` version satisfies both `nnx`'s API
requirement and the torch-2.4.1 import surface; the 2026-06-15 weekly `smoke-tier-b` cron confirmed
this (`AttributeError: module 'torch' has no attribute 'int1'`). See `README.md` §4 and issue #10
for the full rationale. The committed notebook outputs were produced in a side environment with
`torch >= 2.5` + `torchao >= 0.17`.

This notebook belongs to the "efficient/compressed MLP" family: where §8.5 edits the architecture
and §8.7 drives the weights sparse, quantization keeps the architecture and sparsity fixed and
instead reduces the *bitwidth* of the weights (and, in QAT, the activations).

## 8.8.1 Problem & motivation

Once a model trains, you would often like to deploy it smaller and faster. **Quantization** maps
the trained FP32 weights (and, in some recipes, activations) to lower-bitwidth integers — int8,
int4, or mixed — so the deployed model needs less memory and, on hardware with low-bitwidth kernel
support, runs faster. The catch is accuracy: lower bitwidths approximate the FP32 weight
distribution less faithfully, and at aggressive bitwidths (4-bit) the approximation error can cost
real accuracy.

Two canonical recipes trade off differently:

- **PTQ** (post-training quantization) — quantize an already-trained model in one shot. Cheap (no
  extra training), but accuracy can drop if the weight distribution does not quantize cleanly.
- **QAT** (quantization-aware training) — insert *fake-quant* ops during training so the optimizer
  sees the quantization noise and adapts to it. Slower (a full extra training run), but typically
  recovers more accuracy than PTQ at the same bitwidth.

This notebook exists for two reasons:

1. **First in-repo exercise of `nnx.quantize_int8` and the QAT lifecycle.** The `nnx` release
   ships both recipes via the `torchao` backend; this notebook is the canonical side-by-side demo
   on the same baseline architecture, with size + latency + accuracy measured for direct comparison.
2. **The PTQ-vs-QAT trade-off is the deployment decision.** Measuring all three (FP32 baseline, PTQ
   int8, QAT 8da4w) on one figure makes the Pareto frontier — accuracy vs size vs latency —
   directly visible, and the right operating point depends on which constraint binds at deployment.

The falsifiable hypothesis tested by the notebook is that PTQ int8 weight-only quantization
recovers near-FP32 accuracy at a \(\sim 4\times\) size reduction, while 8da4w QAT reaches the
smallest size at a measurable accuracy cost that a longer training budget would mostly close.

## 8.8.2 Concepts

| Concept | Where it shows up |
|---|---|
| Post-training quantization (PTQ) | `nnx.quantize_int8(fp32_model)` — one-shot, no extra training |
| Quantization-aware training (QAT) | `qat_train_step_factory` + `QATLifecycleCallback` — fake-quant during training |
| INT8 weight-only | PTQ packs `Linear` weights as int8 with per-channel scales; activations stay FP |
| 8da4w (8-bit dynamic act, 4-bit weight) | The QAT config; dominant edge-LLM deployment recipe |
| Fake-quant (straight-through estimator) | Forward quantizes; backward uses the FP gradient (STE) |
| Per-channel scales | One scale per output channel; finer than per-tensor quantization |
| int4 groupsize 32 | 4-bit weights grouped in blocks of 32; hidden widths must divide 32 |
| `QATLifecycleCallback` | `on_train_begin` inserts fake-quant; `on_train_end` converts to truly-quantized |
| Manual-only (CI-excluded) | torchao requires torch>=2.5; repo pins torch==2.4.1 |

The `nnx` surface consumed is: `NNModel`, `NNParams`, `NNModelParams`, `NNTrainParams`,
`NNOptimParams`, `NNDataset`, `Activations`, `Devices`, `Losses`, `Nets`, `Optims`, `set_seed`, and
the quantization primitives `nnx.quantize_int8`, `nnx.qat_train_step_factory`,
`nnx.QATLifecycleCallback`. The `torchao` backend is opt-in via the `nnx[quantize]` extra.

## 8.8.3 Mathematical formulation

**Uniform affine quantization.** A real weight \(w\) is mapped to an integer \(\bar{w}\) in
\(\{0, \dots, 2^b - 1\}\) via a scale \(s\) and (optionally) a zero-point \(z\):

\[
\bar{w} = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{w}{s}\right) + z;\;0,\;2^b-1\right), \qquad
\widehat{w} = s\,(\bar{w} - z) \approx w.
\]

For a weight matrix \(W\), per-channel quantization chooses one scale \(s_c\) per output channel
(column of \(W\)), set to \(s_c = \max_i |W_{ic}| / (2^{b-1}-1)\) for symmetric int8. Per-channel
scales track the per-column dynamic range and are markedly more accurate than a single per-tensor
scale for typical weight distributions.

**PTQ (int8 weight-only).** The trained FP32 weights are quantized once to int8 with per-channel
scales; activations stay FP at runtime. The forward pass dequantizes the int8 weights back to FP
for the matmul. This is cheap (no training) and works well when the weight distribution is
quantization-friendly. `nnx.quantize_int8` returns a new `NNModel` whose `Linear` layers are the
torchao int8-weight-only variants.

**QAT (8da4w).** Training inserts *fake-quant* nodes that simulate the quantization rounding in the
forward pass while the backward pass uses the straight-through estimator (STE) — gradients pass
through the rounding unchanged:

\[
\frac{\partial \widehat{w}}{\partial w} = 1 \quad \text{within the clamp range, else } 0.
\]

This lets the optimizer "see" the quantization noise and adapt the FP weights to minimize its
effect. The `QATLifecycleCallback(qat_config="8da4w")` swaps the `Linear` layers for fake-quant
variants at `on_train_begin` and converts them to truly-quantized variants (real int4 weights, real
int8 dynamic activations) at `on_train_end`. The 8da4w recipe uses 8-bit *dynamic* activations
(quantized per-batch at runtime) and 4-bit weights in groups of 32.

**Groupsize constraint.** The int4 weights are quantized in blocks of `groupsize=32` contiguous
entries, each block with its own scale. A `Linear` weight column whose width does not divide 32
either needs `padding_allowed=True` or fails the QAT preparation step. The notebook picks
`HIDDEN_DIMS=[128, 64]` (both multiples of 32) to avoid this entirely.

The training objective is cross-entropy on the MNIST logits,

\[
\mathcal{L}(z, y) = -\log \frac{e^{z_c}}{\sum_j e^{z_j}},
\]

and the baseline optimizer is Adam with \(\eta = 10^{-3}\), \(\beta = (0.9, 0.999)\), weight decay
\(0\).

## 8.8.4 Architecture

![Feed-forward MLP](../diagrams/img/mlp.png)

The architecture is `Nets.FEED_FWD` (`nnx.FeedFwdNN`): a 784-unit input layer (flattened MNIST),
two hidden layers `[128, 64]` with ReLU, and a 10-unit output layer consumed by softmax +
cross-entropy. The widths are chosen as multiples of 32 so the 8da4w int4 groupsize divides
cleanly.

| Object | `hidden_dims` | Role |
|---|---|---|
| FP32 baseline | `[128, 64]` | Trained 3 epochs from scratch; the accuracy ceiling and size floor |
| PTQ int8 (weight-only) | `[128, 64]` | `quantize_int8(fp32_model)` — one-shot on the baseline |
| QAT 8da4w | `[128, 64]` | Fresh model trained end-to-end with the QAT callback |

The shared contract:

- **Net:** `Nets.FEED_FWD`, **activation:** `Activations.RELU`
- **Loss:** `Losses.CROSS_ENTROPY`
- **Optimizer:** Adam, `max_lr=1e-3`, `weight_decay=0.0`, `momentum=(0.9, 0.999)`
- **Device:** `Devices.CPU`
- **Budget:** `N_EPOCHS=3` for both the FP32 baseline and the QAT model
  (or `SMOKE_TEST_EPOCHS=1` under `SMOKE_TEST=1`)
- **Batching:** `batch_sizes=(128, None, None)` — 128-sample train minibatches; val as one batch
- **Seed:** `0`
- **QAT config:** `qat_config="8da4w"` (8-bit dynamic activations, 4-bit weights, groupsize 32)

The *a priori* expectation at MNIST scale + a short budget: PTQ int8 shrinks the state-dict at a
small accuracy hit (possibly *slower* on CPU because torchao dispatch overhead dominates the math
savings at this tiny scale); QAT 8da4w has the smallest converted state-dict but the highest
accuracy cost since 4-bit is aggressive, with recovery partial at 3 epochs.

## 8.8.5 Code walkthrough

### PTQ — one-shot `quantize_int8`

```python
fp32_model = make_model()
fp32_run = fp32_model.train(params=train_params())   # 3 epochs

ptq_model = nnx.quantize_int8(fp32_model)
print(f"PTQ model type: {type(ptq_model).__name__}")  # NNModel
```

`nnx.quantize_int8` takes the trained `NNModel` and returns a *new* `NNModel` whose `Linear` weights
are packed int8 with per-channel scales; activations stay FP. The contract is "same forward shape,
possibly different output values within a quantization tolerance." No extra training is involved —
this is the cheap, one-shot path.

### QAT — lifecycle callback + step factory

```python
qat_model = make_model()
qat_cb = nnx.QATLifecycleCallback(qat_config="8da4w")
qat_step = nnx.qat_train_step_factory(qat_config="8da4w")

qat_run = qat_model.train(
    params=train_params(),
    callbacks=[qat_cb],
    train_step_fn=qat_step,
)
print(f"QAT callback: is_prepared={qat_cb.is_prepared}, is_converted={qat_cb.is_converted}")
```

Three moving parts. The `QATLifecycleCallback` hooks the train loop: at `on_train_begin` it swaps
the model's `Linear` layers for fake-quant variants (the forward quantizes, the backward uses STE);
at `on_train_end` it converts those fake-quant variants to *truly-quantized* variants (real int4
weights, real int8 dynamic activations). The `qat_train_step_factory` returns a custom train step
that knows how to step the fake-quant parameters. Both are passed to `model.train(...)`. The
post-call assertions — `is_prepared=True`, `is_converted=True`, and the presence of
`Int8DynActInt4WeightLinear` modules in `qat_model.net.modules()` — verify the lifecycle ran to
completion.

### The size + latency measurement

```python
def state_size_bytes(model):
    return len(pickle.dumps(model.net.state_dict()))

def avg_latency_us(model, n_batches=10):
    """Mean per-batch forward latency in microseconds (CPU, eval mode)."""
    # warm up, then time n_batches forward passes on the val loader
    ...
```

`state_size_bytes` pickles the `state_dict()` to a bytes blob and takes its length — a direct proxy
for deployed model size. `avg_latency_us` warms up one batch then times ten forward passes on the
validation loader in `eval()` mode, returning the mean per-batch latency in microseconds. Both are
CPU measurements; the latency number is honest about the CPU context (see pitfalls — torchao
dispatch overhead can dominate at this scale).

### The comparison table

```python
rows = [
    ("FP32 baseline",        fp32_edp.loss, fp32_edp.accuracy, state_size_bytes(fp32_model), avg_latency_us(fp32_model)),
    ("PTQ int8 (weight-only)", ptq_edp.loss, ptq_edp.accuracy, state_size_bytes(ptq_model),  avg_latency_us(ptq_model)),
    ("QAT 8da4w (converted)",  qat_edp.loss,  qat_edp.accuracy,  state_size_bytes(qat_model),  avg_latency_us(qat_model)),
]
```

`model.evaluate(ds.val_loader)` returns an evaluation data point carrying loss and accuracy; the
size and latency helpers complete the three-axis comparison. The verdict sorts the three recipes by
the deployment constraint that matters (memory? latency? accuracy floor?).

## 8.8.6 Results & analysis

On the recorded run (side environment, `torch >= 2.5` + `torchao >= 0.17`; committed outputs
produced under `torch 2.8.0`), the three recipes land as:

| Model | Val loss | Val acc | State size (KB) | Fwd latency (µs/batch) |
|---|---|---|---|---|
| FP32 baseline | 2.0587 | 53.48% | 429.3 | 1382 |
| PTQ int8 (weight-only) | 2.0587 | 53.38% | 112.6 | 1721 |
| QAT 8da4w (converted) | 2.0729 | 44.53% | 406.7 | 3512 |

Three observations:

1. **PTQ int8 recovers near-FP32 accuracy at \(\sim 4\times\) size reduction.** Val accuracy drops
   only 0.10 pp (53.48% → 53.38%) while the state-dict shrinks from 429.3 KB to 112.6 KB. This is
   the cheap default working as advertised: int8 weight-only quantization on a friendly
   distribution is essentially free at MNIST scale.
2. **QAT 8da4w is the smallest at the highest accuracy cost.** The converted state-dict is 406.7 KB
   — *larger* than PTQ int8 here, because the 8da4w module wraps the int4 weights with per-group
   scales/zero-points that, at this tiny model width, outweigh the 4-bit savings; on a real LLM
   the 4-bit weight compression dominates and 8da4w is much smaller than int8. Accuracy drops to
   44.53% (a 9-point cost) because 4-bit is aggressive and the 3-epoch budget leaves QAT recovery
   partial — longer schedules typically close most of the gap.
3. **On CPU at MNIST scale, quantization is *slower*, not faster.** PTQ int8 latency is 1721 µs
   (vs FP32's 1382 µs) and QAT 8da4w is 3512 µs. The torchao dispatch overhead dominates the math
   savings at this tiny model size on CPU; the latency win shows up on bigger models, GPUs, and
   mobile NPUs with native low-bitwidth kernels. Read the latency numbers as "CPU-bound, tiny
   model," not as a general statement about quantization speedup.

The right reading: **quantization is a real Pareto trade-off**, and the right operating point
depends on which constraint binds at deployment. PTQ is the cheap default; QAT is the recourse when
PTQ accuracy is not acceptable and a training budget is available. At MNIST scale the size win is
real (PTQ) and the latency win is absent (CPU-bound); both generalize differently to larger models
and accelerator hardware.

## 8.8.7 Pitfalls & edge cases

- **Manual-only — does not run in CI or the recommended genai-vanilla runtime.** This is the
  load-bearing pitfall. `torchao >= 0.9.0` (the earliest version exposing the
  `Int8WeightOnlyConfig` API `nnx.quantize_int8` calls) references `torch.int1` at import time.
  `torch.int1` was added in `torch 2.5`; ml-eng-lab pins `torch==2.4.1` for genai-vanilla image
  parity. **No torchao version satisfies both nnx's API requirement and the torch-2.4.1 import
  surface**, so the notebook cannot execute under CI's pinned environment. The 2026-06-15 weekly
  `smoke-tier-b` cron confirmed this (`AttributeError: module 'torch' has no attribute 'int1'`).
  To run locally, use a side environment with `torch >= 2.5` and `torchao >= 0.17`. See
  [issue #10](https://github.com/thekaveh/ml-eng-lab/issues/10) for full context. The committed
  notebook outputs were produced under `torch 2.8.0`.
- **8da4w is aggressive (4-bit weights).** At the short training budget used here for CPU
  feasibility (3 epochs), QAT recovery is partial (44.53% vs FP32's 53.48%). Longer schedules
  typically close most of the gap; do not read the recorded QAT accuracy as the achievable 8da4w
  ceiling.
- **CPU latency is misleading at MNIST scale.** The torchao dispatch overhead can make int8 and
  8da4w *slower* than FP32 on CPU at this model size (1721 µs and 3512 µs vs 1382 µs). The latency
  win shows up on bigger models, GPUs, and mobile NPUs with native low-bitwidth kernels. Do not
  report the CPU latency as a quantization speedup.
- **Hidden widths must divide the int4 groupsize (32).** The 8da4w default int4 groupsize is 32.
  Hidden widths that do not divide 32 either trigger `padding_allowed=True` (the `nnx` test suite
  uses this) or fail the QAT preparation step. The notebook picks `[128, 64]` (both multiples of
  32) to dodge this completely; an arbitrary width will surprise you.
- **The QAT val-loss is measured on the FP-shadow, not the converted net.** During training the
  fake-quant layers keep an FP "shadow" copy for the loss computation, so `qat_run.idps[-1].val_edp.loss`
  reflects the FP-shadow forward, not the truly-quantized forward. The post-conversion accuracy
  comes from `qat_model.evaluate(...)` *after* `model.train(...)` returns (i.e. after
  `on_train_end` converted the layers). The notebook prints both and they differ slightly — do not
  conflate them.
- **Random-init baseline.** The FP32 baseline is trained from scratch in this notebook (3 epochs).
  The PTQ + QAT delta is measured against this very-short FP32 ceiling; in production you would PTQ
  a more-converged model and the absolute accuracy numbers would be much higher across the board.
  The *shape* of the trade-off is what generalizes; the absolute numbers are a budget artifact.
- **Deprecation noise.** `torchao` emits a `TorchAODType is deprecated, please use torch.intN`
  `UserWarning` at import time under newer torch versions; this is harmless and does not affect
  correctness. Do not treat it as a failure.

## 8.8.8 Extensions & references

- **Lengthen the budget to close the QAT recovery gap.** Train the FP32 baseline to convergence
  (10–20 epochs), then PTQ it; separately train the QAT model for the same total budget. The PTQ
  accuracy hit should shrink (friendlier weight distribution) and the QAT recovery should close
  most of the gap to FP32 — the canonical production outcome the short Tier-A-style budget cannot
  show.
- **Add an int4 PTQ point.** `torchao` supports int4 weight-only PTQ as well; adding it to the
  comparison table gives a fourth operating point (cheaper than QAT, lower-accuracy than QAT in
  expectation) and completes the bitwidth Pareto curve.
- **Measure latency on a GPU / mobile NPU.** The CPU latency numbers here are dispatch-bound and
  misleading. Re-running the latency measurement on a CUDA GPU or a mobile NPU with native int8 /
  int4 kernels exposes the real speedup that motivates quantization in deployment.
- **Pair with pruning (§8.7).** Prune to \(s = 0.7\), then PTQ or QAT the pruned model. Sparse +
  int8 / int4 is the dominant edge-deployment recipe; the joint Pareto curve over the
  \(s \times\) bitwidth grid is the full deployment study.
- **Try `padding_allowed=True` on non-32-multiple widths.** If a wider architecture (e.g.
  `[200, 100]`) is desired, enable padding in the QAT config rather than redesigning widths around
  the groupsize. This trades a small accuracy noise for architectural freedom.
- **References.** Jacob et al., 2018, *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference* (arXiv:1712.05877) — the canonical QAT formulation with the
  straight-through estimator. The `torchao` library documentation covers the `8da4w` recipe
  (`Int8DynActInt4WeightLinear`) and the `Int8WeightOnlyConfig` PTQ path. See also the task
  [`README.md`](../../notebooks/quantization-mnist-ffnn-pytorch/README.md) §4 and
  [`docs/env-setup.md`](../env-setup.md) §6 for the manual-only execution path and the torch/torchao
  pin rationale, and [issue #10](https://github.com/thekaveh/ml-eng-lab/issues/10) for the
  CI-exclusion decision.
