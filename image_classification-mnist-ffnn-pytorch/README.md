# image_classification-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Image classification.
- **Dataset:** MNIST handwritten digits.
- **Model:** Feed-forward neural network — using `nnx.FeedFwdNN`.
- **Framework:** PyTorch (via [`nnx`](../nnx)).

## 2. Why this exists

The PyTorch counterpart to the from-scratch NumPy sibling. Same task, same dataset; this version uses the `nnx` toolkit to demonstrate how the library's training loop, dataset abstractions, and visualization helpers compose into a clean notebook.

This is the canonical reference for "how to build a small classifier using nnx".

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/image_classification-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — task, dataset, approach, libraries.
- §2 Environment & Setup — `nnx` and torchvision imports, hyperparameters, seed/device setup.
- §3 Data — construct an `NNDataset` wrapping torchvision's MNIST.
- §4 Model — `NNModelParams` config (loss, optimizer, scheduler, device); `NNParams` network config; `NNModel` instantiated with `Nets.FEED_FWD`. Architecture rationale.
- §5 Training — `nnx` training loop tracks per-iteration metrics into `NNIterationDataPoint`.
- §6 Evaluation & Results — test-set evaluation into `NNEvaluationDataPoint`; convergence + confusion matrix via `nnx.vis_utils`.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open notebook.ipynb in VS Code (attached to the running container) or browser.
# Run all cells.
```

Or via the Tier-B smoke target (writes to `/tmp/`, preserves committed outputs):

```bash
make smoke-tier-b
```

**Tier-B** (heavier; the full `[9 hidden_dims × 500 epochs]` sweep takes >90 min on the Linux GH runner, see issue #7). Runs in CI only on the weekly schedule (`0 7 * * 1`) or on PRs labeled `tier-b-smoke`, not on every PR. For local re-execution that refreshes committed outputs, open the notebook in Jupyter and run all cells (the make target writes to `/tmp/` to preserve the existing outputs).

Also verified via [`tests/nnx_surface/test_image_classification_mnist_ffnn_pytorch.py`](../tests/nnx_surface/test_image_classification_mnist_ffnn_pytorch.py) — a fast NNx-surface contract test pinning the `NNModel` + `Nets.FEED_FWD` call shape. Runs in the CI `pytest-nnx-surface` job on every PR (`make test-nnx-surface` locally).

## 5. Dependencies

- `nnx` (the submodule)
- `torch` (≥ 2.0)
- `torchvision`
- `matplotlib` (via `nnx.vis_utils` — loss curves, confusion matrices)

All installed by the genai-vanilla jupyterhub image or via the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- `./data/` and `./runs/` are gitignored; first run downloads MNIST and creates a fresh runs directory.
- If you see `ModuleNotFoundError: No module named 'nnx'`, your jupyterhub image was built before genai-vanilla PR #26 (`cbad341`). Rebuild from a current genai-vanilla `main` (`docker compose build jupyterhub`), or run the §2 wrapper-and-bind-mount path with `../scripts/setup-in-jupyter.sh` for the editable-install override. See [`../docs/jupyterhub-integration.md`](../docs/jupyterhub-integration.md) §6.

## 7. Future work

- Add a CNN variant (LeNet-style) for direct comparison against the FFN.
- Add an early-stopping callback example using `nnx` once that primitive lands upstream.
