# image_classification-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Image classification.
- **Dataset:** MNIST handwritten digits.
- **Model:** Feed-forward neural network — using `nnx.FeedFwdNN`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)).

## 2. Why this exists

The PyTorch counterpart to the from-scratch NumPy sibling. Same task, same dataset; this version uses the `nnx` toolkit to demonstrate how the library's training loop, dataset abstractions, and visualization helpers compose into a clean notebook.

This is the canonical reference for "how to build a small classifier using nnx".

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/image_classification-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — task, dataset, approach, libraries.
- §2 Environment & Setup — `nnx` and torchvision imports, hyperparameters, seed/device setup.
- §3 Data — construct an `NNDataset` wrapping torchvision's MNIST.
- §4 Model — `NNModelParams` config (loss, optimizer, scheduler, device); `NNParams` network config; `NNModel` instantiated with `Nets.FEED_FWD`. Architecture rationale.
- §5 Training — `nnx` training loop tracks per-iteration metrics into `NNIterationDataPoint`.
- §6 Evaluation & Results — test-set evaluation into `NNEvaluationDataPoint`; convergence + confusion matrix via `nnx.vis_utils`.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# Open notebook.ipynb in VS Code (attached to the running container) or browser.
# Run all cells.
```

Or via the Tier-B smoke target (writes to `/tmp/`, preserves committed outputs):

```bash
make smoke-tier-b
```

**Tier-B** (heavier; the full `[9 hidden_dims × 2 dropouts × 500 epochs]` sweep takes >90 min on the Linux GH runner, see issue #7). `make smoke-tier-b` writes a **`SMOKE_TEST=1` reduced sweep** (`[1 hidden_dims × 1 dropout × 2 epochs]`, ~5 min on CPU) to `/tmp/` to preserve the committed full-sweep outputs. Runs in CI only on the weekly schedule (`0 7 * * 1`), on `workflow_dispatch`, or on PRs labeled `tier-b-smoke`, not on every PR. For local re-execution that refreshes the committed full-sweep outputs, open the notebook in Jupyter and run all cells.

Also verified via [`tests/nnx_surface/test_image_classification_mnist_ffnn_pytorch.py`](../../tests/nnx_surface/test_image_classification_mnist_ffnn_pytorch.py) — a fast NNx-surface contract test pinning the `NNModel` + `Nets.FEED_FWD` call shape. Runs in the CI `pytest-nnx-surface` job on every PR (`make test-nnx-surface` locally).

## 5. Dependencies

- `nnx` (PyPI: `thekaveh-nnx`)
- `torch` (≥ 2.0)
- `torchvision`
- `matplotlib` (via `nnx.vis_utils` — loss curves, confusion matrices)

All installed by the genai-vanilla jupyterhub image or via the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- `./data/` and `./runs/` are gitignored; first run downloads MNIST and creates a fresh runs directory.
- If you see `ModuleNotFoundError: No module named 'nnx'`, your jupyterhub image was built before genai-vanilla PR #26 (`cbad341`) or against the now-defunct `nnx-pytorch[lm]` PyPI distribution name (the bake-name bump to `thekaveh-nnx[lm]==0.2.0` is the outstanding follow-up to the 2026-06-14 PyPI migration). Rebuild from a current genai-vanilla `main` (`docker compose build jupyterhub`) once upstream picks up the new dep, or run a per-session `docker exec -it <jupyterhub> pip install thekaveh-nnx[lm]==0.2.0` workaround. See [`../docs/jupyterhub-integration.md`](../../docs/jupyterhub-integration.md) §6.

## 7. Future work

- Add a CNN variant (LeNet-style) for direct comparison against the FFN.
- Add an early-stopping callback example using `nnx` once that primitive lands upstream.
