"""NNx-surface contract test for notebooks/tabular_classification-iris-mlp-pytorch/notebook.ipynb.

The iris notebook is the first in-repo use of `NNTabularDataset` (the
pandas-DataFrame → DataLoader wrapper that landed in the 21-commit
nnx hop) alongside the same `FeedFwdNN` + `NNModel` core used in
`image_classification-mnist-ffnn-pytorch`.

If this test fails after a `thekaveh-nnx` version upgrade, the bump
broke something the iris notebook depends on. The test asserts the
*shape* of the call chain (build datasets via DataFrame → build params
→ instantiate model → 1-epoch train → predict → unpack PredictResult →
render a confusion matrix), not the convergence behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nnx import (
    Devices,
    Losses,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTabularDataset,
    NNTrainParams,
    Nets,
    Optims,
    VisUtils,
    set_seed,
)


_FEATURE_COLS = ["f0", "f1", "f2", "f3"]
_TARGET_COL = "y"


@pytest.fixture
def tiny_iris_like_df() -> pd.DataFrame:
    """A 12-row DataFrame shaped like the post-scaling iris frame.

    Three classes × four features, label-balanced, deterministic via a
    seeded rng. Used to exercise NNTabularDataset's DataFrame path.
    """
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for class_idx in range(3):
        for _ in range(4):
            row = {col: float(rng.random()) for col in _FEATURE_COLS}
            row[_TARGET_COL] = class_idx
            rows.append(row)
    return pd.DataFrame(rows)


def test_nntabulardataset_builds_loaders_from_dataframe(tiny_iris_like_df):
    """NNTabularDataset wraps a DataFrame into train/val/test loaders.

    Pins the call shape (df + feature_cols + target_col + per-split
    batch_sizes + val/test proportion) so the iris notebook's §3 build
    keeps working across nnx bumps.
    """
    ds = NNTabularDataset(
        df=tiny_iris_like_df,
        feature_cols=_FEATURE_COLS,
        target_col=_TARGET_COL,
        batch_sizes=(4, None, None),
        val_proportion=0.25,
        test_proportion=0.25,
        name_override="tiny-iris",
    )
    assert ds.input_dim == len(_FEATURE_COLS)
    assert ds.output_dim == 3
    assert ds.name == "tiny-iris"
    assert ds.train_loader is not None
    assert ds.val_loader is not None
    assert ds.test_loader is not None
    # State() snapshot is the rendering surface for repr() / verifier tables.
    snap = ds.state()
    assert snap["input_dim"] == len(_FEATURE_COLS)
    assert snap["output_dim"] == 3


def test_feedfwd_iris_full_pipeline(tiny_tabular_batch):
    """End-to-end smoke: set_seed → NNModel → train one epoch → predict → unpack.

    Uses the shared tiny_tabular_batch fixture (4 samples × 4 features ×
    3 classes) from tests/nnx_surface/conftest.py — same fixture every
    tabular pytest module shares.
    """
    set_seed(0)
    model = NNModel(
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[],  # linear baseline — the notebook's Candidate A
            input_dim=4,
            output_dim=3,
        ),
    )
    train_params = (
        NNTrainParams(
            n_epochs=1,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4, momentum=(0.9, 0.999)),
            seed=0,
        )
        .with_train_loader(value=tiny_tabular_batch.train_loader)
        .with_val_loader(value=tiny_tabular_batch.val_loader)
    )
    run = model.train(params=train_params)
    assert run is not None, "NNModel.train() must return an NNRun"

    result = model.predict(X=tiny_tabular_batch.X)
    logits, classes = result  # PredictResult unpacks as (logits, classes)
    assert logits.shape == (4, 3), f"expected logits.shape == (4, 3), got {logits.shape}"
    assert classes.shape == (4,), f"expected classes.shape == (4,), got {classes.shape}"
    assert np.issubdtype(np.asarray(classes).dtype, np.integer), (
        f"classes must be int-typed, got {np.asarray(classes).dtype}"
    )


@pytest.mark.parametrize(
    "hidden_dims, dropout_prob",
    [
        ([], 0.0),       # Candidate A — linear baseline
        ([8], 0.1),      # Candidate B — 1 hidden
        ([16, 8], 0.1),  # Candidate C — 2 hidden
    ],
)
def test_each_candidate_topology_constructs(tiny_tabular_batch, hidden_dims, dropout_prob):
    """Every §4 candidate topology must construct + 1-epoch train without raising."""
    set_seed(0)
    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=dropout_prob,
            hidden_dims=hidden_dims,
            input_dim=4,
            output_dim=3,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1, optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4, momentum=(0.9, 0.999)))
        .with_train_loader(value=tiny_tabular_batch.train_loader)
        .with_val_loader(value=tiny_tabular_batch.val_loader)
    )
    model.train(params=train_params)  # must not raise


def test_visutils_confusion_matrix_renders_for_tabular_predictions(tiny_tabular_batch):
    """VisUtils.confusion_matrix accepts the (Y_true, Y_pred) arrays the
    iris §6 cell passes in (1-D integer arrays, optional class_names).
    """
    set_seed(0)
    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(dropout_prob=0.0, hidden_dims=[], input_dim=4, output_dim=3),
    )
    train_params = (
        NNTrainParams(n_epochs=1, optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4, momentum=(0.9, 0.999)))
        .with_train_loader(value=tiny_tabular_batch.train_loader)
        .with_val_loader(value=tiny_tabular_batch.val_loader)
    )
    model.train(params=train_params)
    _, classes = model.predict(X=tiny_tabular_batch.X)
    fig = VisUtils.confusion_matrix(
        Y_true=tiny_tabular_batch.y,
        Y_pred=np.asarray(classes),
        class_names=["setosa", "versicolor", "virginica"],
        title="test-only cm",
        normalize=False,
    )
    assert fig is not None  # matplotlib.figure.Figure
