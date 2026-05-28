"""Shared fixtures for NNx-surface tests.

Each fixture produces a *tiny* synthetic batch with no external downloads,
so the whole suite runs in seconds on CPU. The fixtures define the data
contract that every per-notebook test in this directory consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx


@pytest.fixture(autouse=True)
def _nnx_run_isolation(monkeypatch, tmp_path):
    """Autouse: chdir each test into tmp_path and pin global seeds for determinism."""
    monkeypatch.chdir(tmp_path)
    nnx.set_seed(0)


@dataclass(frozen=True)
class TinyTabularBatch:
    """4 samples, 4 features, 3 classes — for FeedFwdNN classification."""

    X: np.ndarray
    y: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader


@dataclass(frozen=True)
class TinyImageBatch:
    """4 samples, 1×28×28 images, 10 classes — for FeedFwdNN on flattened MNIST shape."""

    X: np.ndarray
    y: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader


@dataclass(frozen=True)
class TinyGraphData:
    """6-node PyG graph with random features and a small edge set — for GNN smoke tests."""

    data: object  # torch_geometric.data.Data; typed loosely to avoid hard import at fixture-collection time
    num_features: int
    num_classes: int


def _make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Half/half train/val split, no shuffle — deterministic for assertions."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    n = X.shape[0]
    half = n // 2
    train = TensorDataset(X_t[:half], y_t[:half])
    val = TensorDataset(X_t[half:], y_t[half:])
    return (
        DataLoader(train, batch_size=batch_size, shuffle=False),
        DataLoader(val, batch_size=batch_size, shuffle=False),
    )


@pytest.fixture
def tiny_tabular_batch() -> TinyTabularBatch:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=(4,)).astype(np.int64)
    train_loader, val_loader = _make_loaders(X, y, batch_size=2)
    return TinyTabularBatch(X=X, y=y, train_loader=train_loader, val_loader=val_loader)


@pytest.fixture
def tiny_image_batch() -> TinyImageBatch:
    """28*28-flat tensors (784-d), 10-class labels — feeds the canonical FFNN MNIST shape."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 28 * 28)).astype(np.float32)
    y = rng.integers(0, 10, size=(4,)).astype(np.int64)
    train_loader, val_loader = _make_loaders(X, y, batch_size=2)
    return TinyImageBatch(X=X, y=y, train_loader=train_loader, val_loader=val_loader)


@pytest.fixture
def tiny_graph_data() -> TinyGraphData:
    """Six-node graph with random features. Edge index forms two triangles + bridge.

    Lazy-imports `torch_geometric` inside the fixture so collection-time
    failures (e.g., when PyG wheels aren't installed in a sub-environment)
    are surfaced per-test rather than per-suite.
    """
    from torch_geometric.data import Data

    rng = np.random.default_rng(0)
    num_nodes, num_features, num_classes = 6, 4, 3
    x = torch.from_numpy(rng.standard_normal((num_nodes, num_features)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, num_classes, size=(num_nodes,)).astype(np.int64))
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 3],
            [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5, 3, 2],
        ],
        dtype=torch.long,
    )
    train_mask = torch.tensor([True, True, True, False, False, False])
    val_mask = torch.tensor([False, False, False, True, True, False])
    test_mask = torch.tensor([False, False, False, False, False, True])
    data = Data(
        x=x, y=y, edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )
    return TinyGraphData(data=data, num_features=num_features, num_classes=num_classes)
