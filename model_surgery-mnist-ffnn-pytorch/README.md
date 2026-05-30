# model_surgery-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Function-preserving model surgery (architectural edits to a *trained* model).
- **Dataset:** MNIST handwritten digits (60k train, 10k test, 28×28 grayscale) — same `nnx.NNDataset` wrapper as the sibling `image_classification-mnist-ffnn-pytorch` task.
- **Model:** Feed-forward neural network via `nnx.FeedFwdNN` + `Nets.FEED_FWD`, with **ReLU** activation (`nnx.deepen`'s identity-init insertion is function-preserving only for ReLU — see §6 *Known issues* + the notebook's §4 *Model*).
- **Framework:** PyTorch (via [`nnx`](../nnx)).

## 2. Why this exists

Most training pipelines treat architecture as immutable. Net2Net (Chen et al., 2015) showed a third option: edit a *trained* model's shape (widen a layer, insert a layer, drop a layer) in a way that exactly preserves the forward output, then continue training. The `nnx` v[megamerge] release ships `nnx.widen`, `nnx.deepen`, `nnx.drop_layer`, and `nnx.low_rank_factorize` as a `nnx.surgery` namespace; this notebook is the canonical in-repo demo of the contract and its convergence consequences.

The "function-preservation" contract is the technical headline. Once you have it, warm-start training from a strictly larger model is free — no accuracy cliff at step 0.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/model_surgery-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — Net2WiderNet / Net2DeeperNet, dataset, approach, libraries.
- §2 Environment & Setup — imports, hyperparameters (`BASE_HIDDEN_DIMS=[64, 64]`, `WIDEN_NEW_WIDTH=128`), `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on MNIST (same constants as the sibling pytorch-MNIST task).
- §4 Model — small two-hidden-layer FFN baseline; `nnx.deepen`'s ReLU constraint explained.
- §5 Training — baseline train, then **assert** widen + deepen produce a forward output equal to the original within `atol=1e-5` on a probe batch, then resume training (`continue` vs `warm-widen` vs `cold-widen`).
- §6 Evaluation & Results — comparison table + loss curves; discussion of when warm-start beats cold-start (budget-dependent).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a   # re-runs this notebook plus the pytorch MNIST, GNN phase1, and iris MLP
```

**Tier A** (cheap, ~45 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — MNIST data + tensors.
- `nnx` (the submodule) — `FeedFwdNN`, `NNModel`, `NNDataset`, `widen`, `deepen`, `Activations.RELU`.
- `prettytable` — comparison table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- `nnx.deepen` is function-preserving only for **ReLU**. Sigmoid/tanh/GELU networks would need a different post-insertion init (different bias / weight strategy) to preserve the forward output. The notebook pins the baseline to `Activations.RELU` to honor this contract.
- The "warm-start vs cold-start" comparison is **budget-dependent**: at the very short training budgets used here for CPU feasibility (3 baseline epochs + 5 resume epochs), the cold-start model can sometimes match or beat the warm-start. The Net2Net advantage shows up clearest at *longer* schedules where the cold-start has to spend epochs reaching the warm-start's step-0 loss. The §6.3 cell discusses this directly.
- `nnx.widen` (Net2WiderNet) preserves the forward to numerical precision (`~1e-6`), not bit-exact. `nnx.deepen` (Net2DeeperNet identity-init) IS bit-exact (`0.00`).
