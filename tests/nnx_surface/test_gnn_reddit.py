"""NNx-surface contract test for node_classification-reddit-gnn-pyg/*.

Asserts the canonical call shape used by phase2-notebook4 and (after the
Phase C rewrites of this plan) by every other phase2/phase3 notebook.

`GraphSageNN`, `GraphConvNN`, and `GraphAttNN` all share a `GraphNNBase`
and accept an `NNParams` (with `n_heads` only required for the GAT
branch). This file pins that contract.

The tests don't download Reddit (which is ~1GB). They build a tiny
synthetic PyG `Data` via the `tiny_graph_data` fixture and exercise the
GNN forward pass via a 1-batch `NeighborLoader`.

The GNN forward-pass tests use a 1-batch `NeighborLoader`, which requires
`pyg-lib` or `torch-sparse` at runtime. Both ship Linux/x86 wheels only
(via torch-requirements.txt's find-links URL), so on Darwin/ARM the three
forward-pass tests are skipped cleanly. The `NNParams.state()` round-trip
test runs everywhere - it's pure Python and pins the consolidated-n_heads
contract regardless of PyG state.
"""
from __future__ import annotations

import pytest

from nnx import (
    Devices,
    Losses,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
    Nets,
)


# PyG's NeighborLoader sampler requires either `pyg_lib` (preferred) or
# `torch_sparse` (legacy) at runtime. Wheels for both ship Linux/x86 only
# via the PyG find-links URL in torch-requirements.txt, so they're
# unavailable on Darwin ARM. CI runs on Linux and has them; local Mac
# checkouts skip these tests cleanly.
def _has_pyg_sampler() -> bool:
    try:
        import pyg_lib  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_PYG_SAMPLER = _has_pyg_sampler()
_PYG_SAMPLER_SKIP_REASON = (
    "PyG NeighborLoader requires pyg-lib or torch-sparse at runtime; "
    "wheels only available on linux/x86 via torch-requirements.txt find-links. "
    "CI installs them; local Darwin/ARM skips. Constructor + n_heads contract "
    "(test_nnparams_state_round_trips_n_heads) still runs."
)


@pytest.fixture
def gnn_loaders(tiny_graph_data):
    """Build train/val NeighborLoaders over the tiny PyG graph."""
    from torch_geometric.loader import NeighborLoader

    data = tiny_graph_data.data
    train_loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=int(data.train_mask.sum()),
        input_nodes=data.train_mask,
        shuffle=False,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=int(data.val_mask.sum()),
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


@pytest.mark.skipif(not _HAS_PYG_SAMPLER, reason=_PYG_SAMPLER_SKIP_REASON)
@pytest.mark.parametrize("net_enum", [Nets.GRAPH_SAGE, Nets.GRAPH_CONV])
def test_gnn_train_one_batch_sage_or_conv(net_enum, tiny_graph_data, gnn_loaders):
    """GraphSAGE and GraphConv share the no-attention path. Both should construct + 1-epoch-train."""
    train_loader, val_loader = gnn_loaders
    model = NNModel(
        params=NNModelParams(net=net_enum, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.25,
            hidden_dims=[8],
            input_dim=tiny_graph_data.num_features,
            output_dim=tiny_graph_data.num_classes,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=train_loader)
        .with_val_loader(value=val_loader)
    )
    run = model.train(params=train_params)
    assert run is not None


@pytest.mark.skipif(not _HAS_PYG_SAMPLER, reason=_PYG_SAMPLER_SKIP_REASON)
def test_gat_consolidates_n_heads_into_nnparams(tiny_graph_data, gnn_loaders):
    """Regression test for the GraphAttNNParams audit miss.

    Pre-extraction, GAT used a distinct GraphAttNNParams class. The
    consolidation merged that into NNParams with an Optional[int] n_heads
    field. This test pins the consolidated shape so any future un-merging
    fails CI rather than the weekly Tier-B/C smoke.
    """
    train_loader, val_loader = gnn_loaders
    model = NNModel(
        params=NNModelParams(net=Nets.GRAPH_ATT, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            n_heads=2,
            dropout_prob=0.25,
            hidden_dims=[8],
            input_dim=tiny_graph_data.num_features,
            output_dim=tiny_graph_data.num_classes,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=train_loader)
        .with_val_loader(value=val_loader)
    )
    run = model.train(params=train_params)
    assert run is not None


def test_nnparams_state_round_trips_n_heads(tiny_graph_data):
    """NNParams.state() omits n_heads when None; includes it when set.

    Pins the rule that prevents existing run.id hashes from shifting when
    n_heads-aware models are introduced.
    """
    p_no_heads = NNParams(dropout_prob=0.1, hidden_dims=[8], input_dim=4, output_dim=3)
    p_with_heads = NNParams(n_heads=2, dropout_prob=0.1, hidden_dims=[8], input_dim=4, output_dim=3)
    assert "n_heads" not in p_no_heads.state()
    assert p_with_heads.state()["n_heads"] == 2
