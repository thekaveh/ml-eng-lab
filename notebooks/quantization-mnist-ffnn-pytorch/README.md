# quantization-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Model quantization — post-training (PTQ) vs quantization-aware training (QAT).
- **Dataset:** MNIST handwritten digits via `nnx.NNDataset`, same constants as the sibling pytorch-MNIST task.
- **Model:** `nnx.FeedFwdNN` (`Nets.FEED_FWD`) with `Activations.RELU`; hidden widths `[128, 64]` (multiples of 32 so 8da4w's int4 groupsize fits cleanly).
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)) + `torchao` (PTQ + QAT backend).

## 2. Why this exists

Quantization shrinks deployed models and speeds inference, at some accuracy cost. The two canonical recipes have very different trade-offs:

- **PTQ** (`nnx.quantize_int8`) is one-shot. Cheap. Sometimes hurts accuracy badly.
- **QAT** (`nnx.qat_train_step_factory` + `nnx.QATLifecycleCallback` with `qat_config="8da4w"`) bakes the quantization noise into training so the optimizer adapts. Slower, usually higher accuracy at the same bitwidth.

This notebook is the in-repo demo of both, side by side on the same baseline architecture, with size + latency + accuracy measured for direct comparison.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — PTQ vs QAT contracts, dataset, libraries.
- §2 Environment & Setup — imports, hyperparameters (`HIDDEN_DIMS=[128, 64]`, multiples of 32 for clean int4 groupsize), `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on MNIST.
- §4 Model — FP32 baseline architecture; PTQ + QAT contracts spelled out.
- §5 Training — train FP32 baseline, run `nnx.quantize_int8` on it, separately train a QAT model end-to-end via the lifecycle callback. Asserts the QAT callback advanced through `is_prepared` + `is_converted` and that the converted net contains `Int8DynActInt4WeightLinear` modules.
- §6 Evaluation & Results — comparison table: val loss, val accuracy, pickled state-dict size (KB), per-batch CPU forward latency (µs).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or manually (the only path in ml-eng-lab's pinned env — see the next paragraph for why):

```bash
cd notebooks/quantization-mnist-ffnn-pytorch
papermill --kernel python3 notebook.ipynb /tmp/out.ipynb
```

**Manual-only** (was Tier-B from 2026-06-02 to 2026-06-16, then removed from `Makefile` TIER_B after the weekly cron failed). `torchao>=0.9.0` (the earliest version exposing the `Int8WeightOnlyConfig` API `nnx.quantize_int8` calls) references `torch.int1` at import time. `torch.int1` was added in `torch 2.5`. ml-eng-lab pins `torch==2.4.1` for genai-vanilla image-parity. **No torchao version satisfies both nnx's API requirement AND the torch 2.4.1 import surface**, so the notebook cannot execute under CI's pinned environment regardless of tier. The 2026-06-15 weekly `smoke-tier-b` cron confirmed this (`AttributeError: module 'torch' has no attribute 'int1'`). To run locally, install `torch>=2.5` + `torchao>=0.17` in a side-env. See [issue #10](https://github.com/thekaveh/ml-eng-lab/issues/10) for full context. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — MNIST + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNDataset`, `quantize_int8`, `qat_train_step_factory`, `QATLifecycleCallback`.
- `torchao>=0.17` — PTQ + QAT backend. Opt-in via `nnx[quantize]`; the surface notebook degrades gracefully (skipped tests) when it isn't installed.
- `prettytable` — comparison table rendering.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Bitwidth caveat:** 8da4w is aggressive (4-bit weights). At the short training budget used here for CPU feasibility (3 epochs), QAT recovery is partial. Longer schedules typically close the gap.
- **CPU latency:** at MNIST scale the torchao dispatch overhead can make int8 and 8da4w *slower* than FP32 on CPU. The win shows up on bigger models / GPUs / mobile NPUs.
- **Width constraint:** the 8da4w default int4 groupsize is 32. Hidden widths that don't divide 32 either trigger `padding_allowed=True` (the test suite uses this) or fail the QAT preparation step. We pick `[128, 64]` to dodge that completely.
- **`SMOKE_TEST=1`:** runs a single-epoch baseline + single-epoch QAT; the comparison table is qualitatively similar but at lower accuracy.
- **Random-init baselines:** the FP32 baseline is trained from scratch in this notebook (3 epochs). The PTQ + QAT delta vs that very-short FP32 ceiling is what's measured; in production you'd PTQ a more-converged model and the absolute accuracy numbers would be much higher across the board.
