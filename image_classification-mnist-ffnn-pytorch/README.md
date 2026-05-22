# image_classification-mnist-ffnn-pytorch

**Task:** Image classification.
**Dataset:** MNIST handwritten digits.
**Model:** Feed-forward neural network — using `nnx.nn.net.FeedFwdNN`.
**Framework:** PyTorch (via [`nnx`](../nnx)).

## Why this exists

The PyTorch counterpart to the from-scratch NumPy sibling. Same task, same dataset; this version uses the `nnx` toolkit to demonstrate how the library's training loop, dataset abstractions, and visualization helpers compose into a clean notebook.

This is the canonical reference for "how to build a small classifier using nnx".

## What you'll see in the notebook

- Construct an `NNDataset` wrapping torchvision's MNIST.
- Define an `NNModelParams` config (loss, optimizer, scheduler, device).
- Define an `NNParams` network config and instantiate `NNModel` with `Nets.FeedFwd`.
- Train; the loop tracks per-iteration metrics into `NNIterationDataPoint` objects.
- Evaluate on the test set; collect into `NNEvaluationDataPoint`.
- Visualize convergence and confusion matrix via `nnx.vis_utils`.

## How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open notebook.ipynb in VS Code (attached to the running container) or browser.
# Run all cells.
```

Or via the Tier-A make target:

```bash
make run-tier-a
```

## Tier

**Tier A** (cheap, <5 min on CPU). Re-executed in CI.

## Dependencies

This notebook uses `nnx` (the submodule), `torch`, `torchvision`, `plotly`. All installed by the jupyterhub image (Phase 1 merge `cb4d8f4`) or via the root `requirements.txt` + `torch-requirements.txt`.

## Known issues

- `./data/` and `./runs/` are gitignored; first run downloads MNIST and creates a fresh runs directory.
- If you see `ModuleNotFoundError: No module named 'nnx'` after a fresh container instance, run `../scripts/setup-in-jupyter.sh` once.
