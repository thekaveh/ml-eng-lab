# NNx submodule findings

Issues found by the verify_repo.py loop in the `./nnx` submodule. These are
NOT fixed by this loop (per spec §1.3); they are surfaced here for an upstream
PR follow-up to [thekaveh/NNx](https://github.com/thekaveh/NNx).

## 1. Findings

### 1.1. `NNDataset` default `batch_size` packs the whole train set into one batch

Surfaced by: `diffusion-mnist-ddpm-pytorch`, `text_generation-tinyshakespeare-transformer-pytorch`, `moe-fmnist-mixture-of-experts-pytorch`, `self_supervised-fmnist-jepa-pytorch`.

`NNDataset(ds_class=thv.datasets.MNIST, ...)`'s `train_loader` defaults to `batch_size=54000` (the whole 60k train set minus the val carve-off). For full-batch SGD on classifiers this is fine; for **diffusion / MoE / transformer / JEPA / any task that needs many noise- or routing-level samples per epoch**, one batch per epoch is far too few — the train step runs ~1 time per epoch and the loss barely budges.

Each affected notebook works around this with:

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(ds.train_loader.dataset, batch_size=128, shuffle=True)
```

**Suggested upstream fix**: make `NNDataset(... , batch_size=N)` accept an explicit override, or default to a smaller batch (256 or 512) so the train loader is per-batch granular by default. The current default is "whole train set" which is a very surprising default for anything that uses `train_step_fn=`.

### 1.2. `nnx.deepen` is function-preserving only for `Activations.RELU`

Surfaced by: `model_surgery-mnist-ffnn-pytorch`.

`nnx.deepen(net, after_layer_name=...)` inserts an identity-init `Linear` after a target Linear. The identity init only preserves the forward output when the *activation between* the original Linear and the new Linear is ReLU (since `ReLU(I x) == ReLU(x)` for any `x`; sigmoid/tanh/GELU pass non-negative *and* negative values through differently).

On any non-ReLU activation the surgery raises `ValueError: deepen: activation is 'leaky_relu', but identity-init insertion is function-preserving only for ReLU.` at construction.

**Suggested upstream fix**: implement an activation-aware identity init for sigmoid / tanh / GELU (different bias init that makes the forward equivalent), OR document the constraint more prominently in the `deepen` docstring. The current error message is excellent — the constraint just isn't a one-liner to discover before tripping over it.

### 1.3. `NNTabularDataset` coerces targets to `torch.long` (classification-only)

Surfaced by: `tabular_regression-diabetes-mlp-pytorch`.

`NNTabularDataset(... , y_col=...)` hard-codes `y = torch.tensor(..., dtype=torch.long)` in `__post_init__`. This is correct for classification but breaks regression: `Losses.MEAN_SQUARED_ERROR` expects `float32` targets of shape `(N, 1)`.

Regression notebooks must build the DataLoaders manually:

```python
DataLoader(
    TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float().unsqueeze(-1),
    ),
    ...,
)
```

The `NNTabularDataset` docstring already says *"For regression, prefer to construct the DataLoaders yourself"* — so this is documented behavior, not a bug.

**Suggested upstream fix**: add a `NNTabularDataset(task='regression' | 'classification', ...)` mode that conditionally skips the `torch.long` coercion when `task='regression'`. The current docstring already notes the limitation; the API just needs to grow the explicit knob so regression callers don't have to bypass the wrapper entirely.

### 1.4. `EarlyStopping(monitor=...)` default is `"val_edp.error"`, doesn't exist for regression EDPs

Surfaced by: `tabular_regression-diabetes-mlp-pytorch` (documented in §6, not actually exercised in the notebook).

`EarlyStopping`'s default `monitor="val_edp.error"` works for classification (lower error = better). For regression the EDP has `loss` but no `error` field — `monitor="val_edp.loss"` must be passed explicitly. The error message at runtime is clear; the issue is that the default doesn't gracefully degrade.

**Suggested upstream fix**: detect at construction whether the loss is regression-style (MSE, MAE) and default `monitor="val_edp.loss"` in that case.

### 1.5. `NNRun.save()` prints an absolute path, leaking the maintainer's local layout

Surfaced by: 14 of the 21 Tier-A notebooks carrying baked-in output text of the form `Run saved to /Users/kaveh/repos/ml-lab/.claude/worktrees/overnight-cleanup/runs/<hash>` from notebook outputs committed during the megamerge PR #5 build (an earlier worktree path no longer used). The post-merge `image_classification-mnist-ffnn-pytorch/notebook.ipynb` re-execution under the real repo root produced the simpler `/Users/kaveh/repos/ml-lab/<task>/runs/<hash>` form — still absolute, just less stale.

`NNRun.save()` (in the nnx submodule's training infrastructure) emits a confirmation string with the absolute filesystem path of the saved run directory. Two related issues:

1. **Maintainer-local path leak**: any committed notebook output carries the path from whatever machine + worktree last executed it. This is reproducibility noise (the path is meaningless to anyone else) and a minor privacy/security leak (it advertises the maintainer's `$HOME` layout).
2. **No CI normalization on re-run**: next CI Tier-A re-execution will overwrite the leaked path with the GitHub-runner-local path (`/home/runner/work/ml-lab/ml-lab/...`), trading one absolute leak for another — not a fix.

**Suggested upstream fix**: print a path relative to `cwd` (or to the notebook's parent), or just `Run saved to ./runs/<hash>`. Absolute path is fine in the saved metadata JSON; the human-facing print should be relative.

**Workaround for ml-lab**: leave the baked-in outputs as-is — sweeping them now is futile until nnx's print is changed, since the next CI re-run regenerates the cell with a different absolute path. Once nnx is fixed, the next Tier-A papermill batch will normalize all 14 stale outputs in one go.
