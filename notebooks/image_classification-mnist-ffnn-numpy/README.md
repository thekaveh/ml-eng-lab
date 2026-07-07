# image_classification-mnist-ffnn-numpy

## 1. Task summary

- **Task:** Image classification.
- **Dataset:** MNIST handwritten digits (60k train, 10k test, 28×28 grayscale).
- **Model:** Feed-forward neural network — implemented from scratch in NumPy. No PyTorch, no auto-grad. All forward + backward passes hand-coded.
- **Framework:** NumPy.

## 2. Why this exists

A demonstration that the building blocks of a feed-forward classifier (linear layers, parametric ReLU with α=0.01, softmax + cross-entropy loss, mini-batch SGD with backprop) work without any deep-learning framework. Useful for teaching, for personal reference, and as a sanity counterweight to the PyTorch variant in the sibling folder.

This folder does **not** use the shared `nnx` library. It's intentionally standalone.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb) for full rendering.

- §1 Overview — task, dataset, approach, libraries.
- §2 Environment & Setup — imports, hyperparameters, reproducibility.
- §3 Data — MNIST loading and inspection.
- §4 Model — `FeedFwdNN` composed from `LinearLayer`, `ReluLayer`, `SoftmaxCrossEntropyLayer`. Architecture and design rationale.
- §5 Training — mini-batch SGD with hand-rolled backprop; per-iteration metrics tracked into `IterationDataPoint`.
- §6 Evaluation & Results — validation loss/accuracy curves, comparisons across two learning rates (`lr=0.1` vs `lr=0.01`); the architecture (single `784→10` linear layer + PReLU on logits) is held constant.

Supporting modules in this folder:

- `feed_fwd_nn.py` — the network class. Holds an ordered list of layers; forward chains them; backward iterates in reverse with chain-rule gradient propagation.
- `linear_layer.py` — fully-connected layer with Xavier-style init.
- `relu_layer.py` — element-wise parametric ReLU (α=0.01) + its derivative.
- `softmax_cross_entropy_layer.py` — combined for gradient stability.
- `utils.py`, `consts.py`, `funcs.py`, `iteration_data_point.py` — supporting bits (one-hot encoding, batching, metrics tracking).

## 4. How to run

In the recommended runtime (genai-vanilla jupyterhub, see [../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# From an attached VS Code or browser jupyter session:
# Open notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb, run all cells.
```

Or use papermill (the Tier-A target in the root Makefile):

```bash
make run-tier-a   # re-runs this notebook in place, alongside the other Tier-A notebooks (see docs/env-setup.md §6)
```

**Tier-A** (cheap, <5 min on CPU). Re-executed in CI on every PR.

## 5. Dependencies

- `numpy` (≥ 1.24)
- `torchvision` (only for MNIST dataset loading; the model itself is pure NumPy)

All available via the root `requirements.txt` or the genai-vanilla jupyterhub image.

## 6. Known issues

- The MNIST dataset is downloaded into `./data/` on first run; this is gitignored. First run takes a few extra seconds.
- Numerical precision differs slightly from the PyTorch sibling because of different default floats and accumulation order. Not a correctness issue.

## 7. Future work

- Add a second variant with momentum / Adam to compare against vanilla SGD.
- Add a tiny convolutional layer (`ConvLayer`) implemented in the same hand-coded style.
