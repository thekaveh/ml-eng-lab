# pruning-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Weight pruning — magnitude-based sparsity sweep.
- **Dataset:** MNIST handwritten digits via `nnx.NNDataset`.
- **Model:** `nnx.FeedFwdNN` (`Nets.FEED_FWD`), `Activations.RELU`, `hidden_dims=[256, 128]`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)). 2:4 semi-structured sparsity is mentioned but not exercised (requires CUDA + Ampere+).

## 2. Why this exists

Magnitude pruning is the simplest neural-network compression recipe: zero the smallest-|w| fraction of weights in each layer, deploy only the non-zero entries. The trade-off curve — *how aggressive can you go before accuracy collapses?* — is the deployment question worth a notebook.

`nnx.prune.magnitude_prune` ships with a `bake=True` default that writes the zeros into the actual `.weight` tensor and removes PyTorch's `weight_orig` + `weight_mask` reparam, so the post-prune `state_dict()` has the same keys as the unpruned net. This is the deployable form (no inference-time overhead).

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/pruning-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — magnitude pruning intuition, dataset, libraries.
- §2 Environment & Setup — imports, `SPARSITY_LEVELS=[0.1, 0.3, 0.5, 0.7, 0.9]`, `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on MNIST.
- §4 Model — FP32 baseline `[256, 128]` and the `magnitude_prune` contract.
- §5 Training — train baseline, then deep-copy + prune at each sparsity, measure val accuracy + actual zero fraction.
- §6 Evaluation & Results — comparison table + accuracy / zero-fraction-vs-sparsity Pareto curve; discussion of the curve's knee + a note on 2:4 semi-structured sparsity (`nnx.prune.semi_structured_24`, CUDA-only).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~15 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell; `SMOKE_TEST=1` sweeps only `s=0.1`.

## 5. Dependencies

- `torch`, `torchvision` — MNIST + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNDataset`, `nnx.prune.magnitude_prune`.
- `matplotlib` — Pareto curve.
- `prettytable` — comparison table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **2:4 semi-structured sparsity is CUDA-only.** `nnx.prune.semi_structured_24` calls torchao's `SparseSemiStructuredTensor`, which only supports CUDA tensors on Ampere or newer architectures. On CPU the swap raises `RuntimeError` at construction; we don't exercise that path. The notebook's §6.3 discusses the trade-off.
- **Short training budget.** The baseline trains for 3 epochs to keep the notebook Tier-A (~15 s). Absolute accuracy numbers are well below MNIST state-of-the-art; the *shape* of the accuracy-vs-sparsity curve is what's pedagogically interesting, and it's stable across training budgets.
- **`bake=True` default.** If you want to *keep training* the pruned model with the mask preserved (gradient-masking semantics), pass `bake=False` and PyTorch's pruning reparam stays in place. This notebook deploys the baked form.
