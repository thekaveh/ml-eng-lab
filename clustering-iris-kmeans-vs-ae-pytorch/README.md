# clustering-iris-kmeans-vs-ae-pytorch

## 1. Task summary

- **Task:** Unsupervised clustering — KMeans on raw 4-D iris features vs on a 2-D autoencoder latent.
- **Dataset:** Iris (`sklearn.datasets.load_iris`, 150 samples × 4 features × 3 species).
- **Models:** `sklearn.KMeans(n_clusters=3)` on raw features and on the AE latent; AE = `FeedFwdNN(input_dim=4, output_dim=4, hidden_dims=[2])` trained with `nnx.NNModel.train(train_step_fn=autoencoder_step)`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)) + sklearn.

## 2. Why this exists

KMeans's cluster quality depends on the **feature space geometry**. Clusters that look like spheres in the input cluster cleanly; ones that don't, don't. A non-linear autoencoder maps the input into a learned latent where the reconstruction objective implicitly groups similar inputs. KMeans on the AE latent typically beats KMeans on raw features on cluster-vs-true-label agreement metrics (ARI, NMI).

This notebook makes the comparison concrete on iris: train an AE inline (no external checkpoint dependency for CI), recover the 2-D latent, run KMeans on both feature spaces, compare ARI + NMI.

**Sibling to `dim_reduction-iris-autoencoder-pytorch/`** — that notebook does the *supervised linear-probe* benchmark on the same AE latent. This notebook does the *unsupervised KMeans* benchmark. Different evaluation axis, same architectural trick.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/clustering-iris-kmeans-vs-ae-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — KMeans + AE pairing, ARI / NMI definitions, sibling pointer.
- §2 Environment & Setup — imports, hyperparameters (`HIDDEN_DIMS=[2]`, `N_EPOCHS=300`, `N_CLUSTERS=3`), `nnx.set_seed(0)`.
- §3 Data — iris loader, MinMax scaling, single `DataLoader` (no train/val/test split — this is unsupervised; labels used *only* for evaluation).
- §4 Model — AE construction + KMeans / ARI / NMI contracts.
- §5 Training — `autoencoder_step` + train; recover 2-D latent by walking the encoder half.
- §6 Evaluation & Results — comparison table (ARI + NMI on both feature spaces); 2×2 scatter grid (AE latent + PCA-projected raw, truth vs KMeans clusters); §6.3 discussion.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~15 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell; `SMOKE_TEST=1` drops `N_EPOCHS` to 5 — the AE under-trains visibly and the latent's KMeans win shrinks.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNTrainParams`, `train_step_fn`, `NNEvaluationDataPoint`, `set_seed`.
- `scikit-learn` — iris loader, `KMeans`, `adjusted_rand_score`, `normalized_mutual_info_score`, `MinMaxScaler`, `PCA` (for raw-feature 2-D projection in §6.2).
- `matplotlib` — 2×2 scatter grid.
- `prettytable` — comparison table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **AE is retrained inline, not loaded from the sibling.** The original plan for this candidate (in `docs/superpowers/plans/`) called for loading the AE checkpoint produced by `dim_reduction-iris-autoencoder-pytorch/runs/<best>/`. That doesn't work on a fresh CI checkout because `runs/` is gitignored. The inline retrain produces an equivalent AE (same architecture, same seed, same `N_EPOCHS`) — slightly slower than checkpoint-loading would be, but CI-safe.
- **KMeans `n_init=10` random-restarts default**. Without that, KMeans on the AE latent can hit local minima and the win over raw features looks smaller / noisier. We pin `n_init=10` and `random_state=0` for reproducibility.
- **The AE-latent KMeans win shrinks with shorter training budgets**. At `SMOKE_TEST=1` (5 epochs), the AE underfits and its latent ARI ≈ raw-feature ARI. The recorded run with `N_EPOCHS=300` shows the win clearly (ARI 0.716 → 0.835).
- **Labels are used for evaluation only.** ARI / NMI score the agreement between KMeans clusters and the true species labels. The clustering itself is fully unsupervised — KMeans never sees `y`.
