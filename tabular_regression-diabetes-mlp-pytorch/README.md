# tabular_regression-diabetes-mlp-pytorch

## 1. Task summary

- **Task:** Regression — predict a continuous target.
- **Dataset:** sklearn `load_diabetes` (442 patients × 10 numeric features, target = disease-progression score 1 year after baseline).
- **Models:** `sklearn.LinearRegression`, `sklearn.KNeighborsRegressor(k=5)`, `nnx` MLP small `[8]`, `nnx` MLP deep `[32, 16]`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)) + sklearn.

## 2. Why this exists

This is the **first regression task in ml-lab**. The collection's other tasks all do classification (image, tabular, node). Regression is structurally different in three places: 1-output head (not `n_classes`), MSE loss (not cross-entropy), R²/MAE/RMSE metrics (not accuracy/F1). This notebook lands the slot and walks through each difference explicitly so the recipe is reusable for the future regression tasks in the roadmap (anomaly-detection, time-series forecasting, etc.).

The notebook also surfaces a real footgun in the nnx API: **`nnx.NNTabularDataset` doesn't fit regression**. It coerces targets to `torch.long` (it's hard-coded for classification — see its `__post_init__`). For regression we MUST build the DataLoaders manually with `dtype=torch.float32` targets and `shape=(N, 1)` to match the network `output_dim=1` against `Losses.MEAN_SQUARED_ERROR`. The §3.3 cell does this; the README §6 calls it out as a known gotcha for downstream regression notebooks.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/tabular_regression-diabetes-mlp-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — task framing, the NNTabularDataset-vs-regression discovery, libraries.
- §2 Environment & Setup — imports, hyperparameters (`N_EPOCHS=200`, `LR=1e-2`, `BATCH_SIZE=32`), `nnx.set_seed(0)`.
- §3 Data — `load_diabetes`, per-feature describe, 70/15/15 split, manual `DataLoader`s with `float32` targets shaped `(N, 1)`.
- §4 Model — sklearn baselines (LinearRegression + KNeighborsRegressor); `make_mlp(hidden_dims)` builds `FeedFwdNN(input_dim=10, output_dim=1, loss=MEAN_SQUARED_ERROR)`.
- §5 Training — train both MLPs with Adam + weight_decay=1e-4.
- §6 Evaluation & Results — comparison table (MSE / RMSE / MAE / R² on held-out test), predicted-vs-actual and residual scatter plots for all 4 models; §6.3 discussion of why linear regression is hard to beat at this dataset scale.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~22 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNTrainParams`, `Losses.MEAN_SQUARED_ERROR`.
- `scikit-learn` — diabetes loader, `LinearRegression`, `KNeighborsRegressor`, `StandardScaler`, metrics.
- `pandas` — feature describe.
- `matplotlib` — predicted-vs-actual + residual plots.
- `prettytable` — comparison table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **`nnx.NNTabularDataset` is classification-only.** Its `__post_init__` coerces targets to `torch.long`; using it for regression silently truncates fractional targets and then crashes inside the loss computation when the network outputs floats. The notebook's §3.3 builds the loaders manually. Future regression tasks in the roadmap (anomaly-detection autoencoder, time-series forecasting) will need the same workaround until the upstream nnx API gains a regression mode.
- **MLPs lose to LinearRegression at this scale.** R² 0.225 (linear) > 0.159 (MLP small) > 0.130 (MLP deep) > 0.028 (KNN) on the recorded run. Diabetes is a 442-sample classical-statistics benchmark — small enough that linear is well-conditioned, and the extra MLP capacity overfits the 308-sample train split before extracting any non-linear signal. The §6.3 prose owns this.
- **`EarlyStopping`'s default monitor is the wrong one for regression.** Default monitor is `"val_edp.error"` — for *classification* (lower error = better, "min" mode). For regression there is no `error` field on the EDP; the right monitor is `"val_edp.loss"`. We don't use EarlyStopping in this notebook to keep the budget predictable, but downstream regression notebooks should pass `monitor="val_edp.loss"` explicitly if they do.
