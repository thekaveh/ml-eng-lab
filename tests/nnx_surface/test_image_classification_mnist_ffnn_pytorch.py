"""NNx-surface contract test for image_classification-mnist-ffnn-pytorch/notebook.ipynb.

If this test fails after a `thekaveh-nnx` version upgrade, the bump broke
something the MNIST FFNN notebook depends on. Test asserts the *shape*
of the call chain (build params → instantiate model → 1-batch train →
predict), not the convergence behavior.
"""
from __future__ import annotations

import numpy as np

from nnx import (
    Devices,
    Losses,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
    Nets,
)


def test_feedfwd_train_one_batch_and_predict(tiny_image_batch):
    """Build a 784→10 FFNN, train one epoch on the tiny batch, predict on X."""
    model = NNModel(
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
        net_params=NNParams(
            dropout_prob=0.1,
            hidden_dims=[16],
            input_dim=28 * 28,
            output_dim=10,
        ),
    )

    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=tiny_image_batch.train_loader)
        .with_val_loader(value=tiny_image_batch.val_loader)
    )

    run = model.train(params=train_params)
    assert run is not None, "NNModel.train() must return an NNRun"

    result = model.predict(X=tiny_image_batch.X)
    # PredictResult unpacks as (logits, classes) — back-compat with the legacy 2-tuple.
    logits, classes = result
    assert logits.shape == (4, 10), f"expected logits.shape == (4, 10), got {logits.shape}"
    assert classes.shape == (4,), f"expected classes.shape == (4,), got {classes.shape}"
    assert np.issubdtype(classes.dtype, np.integer), f"classes must be int-typed, got {classes.dtype}"


def test_feedfwd_no_hidden_layers(tiny_image_batch):
    """Smoke: hidden_dims=[] is a valid 'logistic-regression' configuration the notebook also uses."""
    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[],
            input_dim=28 * 28,
            output_dim=10,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=tiny_image_batch.train_loader)
        .with_val_loader(value=tiny_image_batch.val_loader)
    )
    model.train(params=train_params)  # must not raise

    logits, classes = model.predict(X=tiny_image_batch.X)
    assert logits.shape == (4, 10)
    assert classes.shape == (4,)
