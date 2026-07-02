"""Regression tests for the from-scratch NumPy task helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_utils_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "image_classification-mnist-ffnn-numpy" / "utils.py"
    spec = importlib.util.spec_from_file_location("mnist_numpy_utils", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Utils


Utils = _load_utils_class()


def test_one_hot_encode_respects_explicit_nonzero_class_order():
    classes = np.array([4, 2])
    labels = np.array([2, 4, 2])

    encoded = Utils.one_hot_encode(labels, classes)

    assert encoded.tolist() == [[0, 1], [1, 0], [0, 1]]


def test_one_hot_encode_accepts_string_labels():
    classes = np.array(["dog", "cat"])
    labels = np.array(["cat", "dog", "cat"])

    encoded = Utils.one_hot_encode(labels, classes)

    assert encoded.tolist() == [[0, 1], [1, 0], [0, 1]]
