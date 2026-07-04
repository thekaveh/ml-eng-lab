# NNx (`thekaveh-nnx`) findings

> **Note (2026-06-14)**: ml-eng-lab switched from a git submodule at `./nnx` to the `thekaveh-nnx` PyPI distribution. Source paths cited below (e.g. `nnx/src/nnx/nn/dataset/nn_dataset.py:24`) refer to the upstream [`thekaveh/NNx`](https://github.com/thekaveh/NNx) repo, not a local submodule.

Issues found by the verify_repo.py loop in the `nnx` (PyPI: `thekaveh-nnx`) library. These are
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

**Upstream fix landed (partial)**: `nnx.NNDataset` now accepts a `batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)` constructor arg (`nnx/src/nnx/nn/dataset/nn_dataset.py:24`), so the cleaner form is `NNDataset(..., batch_sizes=(128, None, None))`. The four affected notebooks still use the older `DataLoader(...dataset, batch_size=128)` bypass; they can be migrated to the `batch_sizes=` form at any time without changing recorded outputs (the resolved batch_size is identical). The default — `None` per slot → whole-split batch — is unchanged upstream, so the underlying "surprising default" critique still stands for new tasks; the workaround just has a less invasive form now.

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

### 1.5. `NNRun.save()` prints an absolute path, leaking the execution environment layout

Surfaced by: historical active notebook outputs carrying baked-in local paths
such as maintainer worktrees, JupyterHub mounts, removed in-repo source trees,
and host-local Python environments. The 2026-07-04 maintenance pass normalized
the remaining active-notebook artifacts and added verifier rule
`E13.stale_active_notebook_path`, so `python scripts/verify_repo.py --check
execution --fast` now rejects stale path artifacts in active notebooks.

`NNRun.save()` (in nnx's training infrastructure) emits a confirmation string with the absolute filesystem path of the saved run directory. Two related issues:

1. **Execution-environment path leak**: any committed notebook output can carry the path from whatever machine, container, or worktree last executed it. This is reproducibility noise because the path is meaningless to readers outside that runtime.
2. **CI normalization is not sufficient**: a CI Tier-A re-execution can replace a local absolute path with a GitHub-runner path, trading one environment-specific artifact for another.

**Suggested upstream fix**: print a path relative to `cwd` (or to the notebook's parent), or just `Run saved to ./runs/<hash>`. Absolute path is fine in the saved metadata JSON; the human-facing print should be relative.

**Workaround for ml-eng-lab**: keep active notebook outputs free of stale
machine-local paths through `E13.stale_active_notebook_path`, and avoid
claiming that a CI re-run alone makes these outputs portable. Once nnx prints a
relative run path, the next Tier-A papermill batch (`make run-tier-a` —
re-executes in place) can refresh outputs without reintroducing environment-
specific paths.
