"""Regression tests for the from-scratch NumPy notebook layers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "image_classification-mnist-ffnn-numpy"
)
sys.path.insert(0, str(NOTEBOOK_DIR))

from linear_layer import LinearLayer  # noqa: E402
from relu_layer import ReluLayer  # noqa: E402
from softmax_cross_entropy_layer import SoftmaxCrossEntropyLayer  # noqa: E402


def test_linear_layer_rejects_supplied_weight_shape():
    with pytest.raises(ValueError, match=r"W\.shape"):
        LinearLayer(
            W=np.ones((3, 2)),
            feature_size_in=3,
            feature_size_out=2,
        )


def test_linear_layer_rejects_supplied_bias_shape():
    with pytest.raises(ValueError, match=r"b\.shape"):
        LinearLayer(
            b=np.ones((2, 1)),
            feature_size_in=3,
            feature_size_out=2,
        )


def test_linear_relu_softmax_layers_run_forward_and_backward():
    linear = LinearLayer(
        W=np.array([[1.0, -1.0], [0.5, 0.25]]),
        b=np.array([0.1, -0.2]),
        feature_size_in=2,
        feature_size_out=2,
    )
    relu = ReluLayer(feature_size=2)
    loss = SoftmaxCrossEntropyLayer(feature_size=2)

    X = np.array([[2.0, 1.0], [-1.0, 3.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    logits = linear.forward(X)
    activations = relu.forward(logits)
    loss_value = loss.forward(activations, Y)
    d_activations = loss.backward()
    d_logits = relu.backward(d_activations)
    dW, db = linear.backward(d_logits)

    assert np.isfinite(loss_value)
    assert dW.shape == (2, 2)
    assert db.shape == (2,)
