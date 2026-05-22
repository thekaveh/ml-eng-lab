# image_classification-mnist-ffnn-numpy

**Task:** Image classification.
**Dataset:** MNIST handwritten digits (60k train, 10k test, 28×28 grayscale).
**Model:** Feed-forward neural network — implemented from scratch in NumPy. No PyTorch, no auto-grad. All forward + backward passes hand-coded.
**Framework:** NumPy.

## Why this exists

A demonstration that the building blocks of a feed-forward classifier (linear layers, ReLU, softmax + cross-entropy loss, mini-batch SGD with backprop) work without any deep-learning framework. Useful for teaching, for personal reference, and as a sanity counterweight to the PyTorch variant in the sibling folder.

## Implementation notes

- `feed_fwd_nn.py` — the network class. Holds an ordered list of layers; forward chains them; backward iterates in reverse with chain-rule gradient propagation.
- `linear_layer.py` — fully-connected layer with Xavier-style init.
- `relu_layer.py` — element-wise ReLU + its derivative.
- `softmax_cross_entropy_layer.py` — combined to keep the gradient stable.
- `utils.py`, `consts.py`, `funcs.py`, `iteration_data_point.py` — supporting bits (one-hot encoding, batching, metrics tracking).

This folder does **not** use the shared `nnx` submodule. It's intentionally standalone.

## How to run

In the recommended runtime (genai-vanilla jupyterhub, see [../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# From an attached VS Code or browser jupyter session:
# Open image_classification-mnist-ffnn-numpy/notebook.ipynb, run all cells.
```

Or use papermill (the Tier-A target in the root Makefile):

```bash
make run-tier-a   # re-runs this notebook plus the pytorch MNIST and GNN phase1
```

## Tier

**Tier A** (cheap, <5 min on CPU). Re-executed in CI on every PR.

## Known issues

- The MNIST dataset is downloaded into `./data/` on first run; this is gitignored. First run takes a few extra seconds.
- Numerical precision differs slightly from the PyTorch sibling because of different default floats and accumulation order. Not a correctness issue.
