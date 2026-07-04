"""NNx-surface contract tests for the manual-only quantization notebook.

The notebook itself cannot run under the repo's pinned Torch 2.4.1 stack because
the required torchao APIs import `torch.int1` from Torch 2.5+. These tests still
guard the public nnx facade the notebook uses, and run a tiny PTQ smoke in
side-envs where the backend is importable.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import nnx
from nnx import (
    Devices,
    Losses,
    NNModel,
    NNModelParams,
    NNParams,
    Nets,
)


def _import_torchao_or_skip():
    if not hasattr(torch, "int1"):
        pytest.skip("torchao quantization path requires torch.int1 from torch >= 2.5")
    return pytest.importorskip("torchao")


def test_quantization_facade_signatures_match_notebook_contract():
    assert inspect.signature(nnx.quantize_int8) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "model",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="NNModel",
            )
        ],
        return_annotation="NNModel",
    )

    qat_sig = inspect.signature(nnx.qat_train_step_factory)
    assert list(qat_sig.parameters) == ["base_step", "qat_config"]
    assert qat_sig.parameters["base_step"].default is None
    assert qat_sig.parameters["qat_config"].default == "8da4w"

    cb_sig = inspect.signature(nnx.QATLifecycleCallback)
    assert list(cb_sig.parameters) == ["qat_config", "groupsize"]
    assert cb_sig.parameters["qat_config"].default == "8da4w"
    assert cb_sig.parameters["groupsize"].default == 32


def test_torchao_guard_skips_before_import_when_torch_lacks_int1(monkeypatch):
    monkeypatch.delattr(torch, "int1", raising=False)

    def fail_importorskip(name):
        raise AssertionError(f"{name} should not be imported without torch.int1")

    monkeypatch.setattr(pytest, "importorskip", fail_importorskip)
    with pytest.raises(pytest.skip.Exception, match="torch.int1"):
        _import_torchao_or_skip()


def test_quantize_int8_predicts_with_same_output_shape_when_backend_available(tiny_image_batch):
    _import_torchao_or_skip()

    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[32],
            input_dim=28 * 28,
            output_dim=10,
        ),
    )

    quantized = nnx.quantize_int8(model)

    logits, classes = quantized.predict(X=tiny_image_batch.X)
    assert quantized is not model
    assert logits.shape == (4, 10)
    assert classes.shape == (4,)
    assert np.issubdtype(classes.dtype, np.integer)


def test_qat_facade_constructs_callback_and_train_step_when_backend_available():
    _import_torchao_or_skip()

    callback = nnx.QATLifecycleCallback(qat_config="8da4w")
    train_step = nnx.qat_train_step_factory(qat_config="8da4w")

    assert callback.qat_config == "8da4w"
    assert callback.groupsize == 32
    assert callable(train_step)
