# dim_reduction-iris-autoencoder-pytorch

## 1. Task summary

- **Task:** Dimensionality reduction — project 4-D iris features to a 2-D latent and visualize species separation.
- **Dataset:** Iris (`sklearn.datasets.load_iris`, 150 samples × 4 features × 3 species).
- **Models:** PCA (sklearn baseline) + two autoencoder variants — shallow `[2]`-bottleneck and deeper symmetric `[3, 2, 3]`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)) + sklearn (PCA + LogisticRegression linear probe).

## 2. Why this exists

PCA is the linear textbook answer for "project 4-D iris into 2-D so we can plot the three species". A non-linear autoencoder is the natural next step: same 2-D latent surface, but learned via reconstruction MSE instead of variance-maximizing eigenvectors. The interesting comparison is which one separates the three species more cleanly — quantified here by a held-out linear-probe accuracy on the recovered latent.

The autoencoder is also the canonical motivating example for `nnx.NNModel.train`'s `train_step_fn` hook. The default supervised forward → `loss_fn(net(X), Y)` → backward path doesn't fit reconstruction — there's no `Y`, the loss is `MSE(decoder(encoder(X)), X)`. The hook lets the notebook swap in its own step body while `NNModel` still owns the scheduler / checkpoint cadence / val loop. **This is the first in-repo demo of `train_step_fn` outside the transformer LM task.**

The other neat trick: a `FeedFwdNN(input_dim == output_dim, hidden_dims=[…])` is *structurally* an autoencoder. No new `nn.Module` subclass, no custom encoder/decoder split — the bottleneck is just the middle Linear layer.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/dim_reduction-iris-autoencoder-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — PCA vs AE, dataset, libraries; framing as the canonical `train_step_fn` demo.
- §2 Environment & Setup — imports, hyperparameters (`N_EPOCHS=300`, `LR=5e-3`, `LATENT_DIM=2`), `nnx.set_seed(0)`.
- §3 Data — iris loader, per-species summary stats, 70/15/15 stratified split, MinMax scaling to `[0, 1]`.
- §4 Model — `make_autoencoder(hidden_dims)` builds `FeedFwdNN(input_dim=4, output_dim=4, hidden_dims=...)`; `train_step_fn` contract.
- §5 Training — `autoencoder_step` (MSE reconstruction, ignore y), train shallow + deeper variants, recover 2-D latents by walking the encoder half manually (`net.layers[:bottleneck_idx + 1]`).
- §6 Evaluation & Results — side-by-side scatter plots of PCA + shallow AE + deeper AE 2-D latents colored by species; LogisticRegression linear-probe accuracy table on the held-out test split; §6.3 discussion of when deeper buys you something (data-dependent — iris is too clean).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~18 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell; `SMOKE_TEST=1` drops `N_EPOCHS` to 5 — the demo still finishes but the AE under-trains visibly.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNTrainParams`, `TrainStepContext`, `NNEvaluationDataPoint`, `set_seed`, `train_step_fn` hook.
- `scikit-learn` — iris loader, PCA, `LogisticRegression` linear probe, `MinMaxScaler`.
- `pandas` — per-species summary stats.
- `matplotlib` — latent scatter plots.
- `prettytable` — linear-probe accuracy table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Linear-probe accuracy can swing run-to-run.** With only 22 test samples, single-class mis-predictions move the accuracy by ~4.5 pp. The qualitative ordering (shallow AE > PCA on this seed) is stable; the absolute numbers aren't.
- **Deeper AE can lose to shallow.** At iris scale (150 samples) the extra capacity in `[3, 2, 3]` often *overfits the reconstruction objective* and loses some species-separation in the bottleneck. The §6.3 prose owns this.
- **No explicit encoder/decoder modules.** We extract latents by walking `net.layers[: bottleneck_idx + 1]` manually with `F.relu` between Linears. This mirrors what `FeedFwdNN.forward` does internally; if you change the activation in `NNParams`, also update the latent-extractor in §5.3.
- **MSE on MinMax-scaled `[0, 1]` inputs.** Switching to `StandardScaler` (mean 0, std 1) would change the absolute reconstruction-loss scale but not the species-separation ranking.

## 7. Sibling: `clustering-iris-kmeans-vs-ae-pytorch/`

The unsupervised-clustering counterpart on the same AE recipe lives at [`../clustering-iris-kmeans-vs-ae-pytorch/`](../clustering-iris-kmeans-vs-ae-pytorch/) — KMeans on raw 4-D features vs on the 2-D AE latent, scored with ARI + NMI vs the true species labels.

Originally this notebook was intended as the *producer* of a saved AE checkpoint at `runs/<best>/` that the sibling would load. In practice `runs/` is gitignored (so the checkpoint isn't available on a fresh CI checkout), and the sibling **retrains the AE inline** with the same architecture / seed / `N_EPOCHS` for CI isolation. No cross-notebook checkpoint dependency at runtime; the two notebooks are independent.
