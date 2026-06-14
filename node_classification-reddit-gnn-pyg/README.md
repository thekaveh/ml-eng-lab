# node_classification-reddit-gnn-pyg

## 1. Task summary

- **Task:** Node classification on a large social-network graph.
- **Dataset:** Reddit2 (PyTorch Geometric). 232,965 nodes, 23,213,838 edges, 602-dim node features, 41 classes (subreddit communities).
- **Models:** Feed-forward NN (baseline), GraphConv (GCN), GraphSAGE, Graph Attention (GAT) — comparison study via PyTorch Geometric.
- **Framework:** PyTorch + PyG (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)).

## 2. Why this exists

A staged investigation of how graph-structural information improves node classification compared to feature-only baselines. The work answers four research questions across three phases of notebooks:

1. How does a traditional Feed-Forward NN perform on node classification with large graphs?
2. Can a Graph Convolutional NN outperform the FFN, and at what computational cost?
3. How does GraphSAGE compare to GCN and the FFN?
4. What does the attention mechanism (GAT) add?

The Phase-3 notebooks preserve August-2023 training outputs and serve as the primary experimental record; the codebase here lets the analysis be re-run on demand.

## 3. What's in the notebook(s)

Nine notebooks across three phases. Each follows the standard 6-section structure (see [../CONTRIBUTING.md](../CONTRIBUTING.md)).

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs (Phase 1 EDA + Phase 2/3 training curves trip the renderer). [Browse this folder on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/tree/main/node_classification-reddit-gnn-pyg/) for full rendering.

### 3.1 Phase 1 — dataset exploration

- **phase1-dataset-exploration-notebook.ipynb** — §1 Overview, §2 Setup, §3 Dataset deep-dive (graph properties, degree distribution, community structure via Louvain, t-SNE projections).

### 3.2 Phase 2 — model selection

Four notebooks sweeping architectures, depths, and learning rates over short training budgets to pick a winner per architecture.

- **phase2-model-selection-notebook1.ipynb** — initial 16-combo sweep (16 combos = 4 architectures × 1 optimizer (Adam) × 2 learning rates × 2 dropouts, `[128]` hidden, 100 epochs).
- **phase2-model-selection-notebook2.ipynb** — 500-epoch convergence study of all four models at consistent hyperparameters.
- **phase2-model-selection-notebook3.ipynb** — deep architecture testing (`[1024, 512, 256]` hidden, 250 epochs; GAT excluded due to GPU-memory constraints).
- **phase2-model-selection-notebook4.ipynb** — 1000-epoch GAT extended training (single architecture, not a sweep; see also phase 3).

### 3.3 Phase 3 — final training and evaluation

Long training runs of the top picks from Phase 2 (Tier-C — preserved outputs, do not re-execute in place; see [../docs/env-setup.md](../docs/env-setup.md)).

- **phase3-main-model-training-and-eval-notebook.ipynb** — GraphAttention (4 heads, hidden `[128]`, 1200 epochs; test acc 0.7660).
- **phase3-main-model-training-and-eval-notebook2.ipynb** — GraphSAGE depth-1 (hidden `[1024, 512, 256, 128]`, 2000 epochs).
- **phase3-main-model-training-and-eval-notebook3.ipynb** — GraphSAGE depth-2 (hidden `[1024, 512, 256, 128, 64]`, 2000 epochs; best overall — val 0.1509, test acc 0.8598).
- **phase3-main-model-training-and-eval-notebook4.ipynb** — GraphSAGE depth-3 (hidden `[768, 1024, 512, 256, 128, 64]`, 2000 epochs; comparable to depth-2 — diminishing returns from added depth).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open any of the notebooks in VS Code attached to the container, or in browser jupyter.
```

| Tier | Notebooks | Command |
|---|---|---|
| **A** | phase1 | `make run-tier-a` (re-runs in place, refreshes outputs) |
| **B** | phase2 × 4 | `make smoke-tier-b` (writes to `/tmp/`, preserves outputs) |
| **C** | phase3 × 4 | `make smoke-tier-c` (writes to `/tmp/`, preserves outputs) |

**Never** run `papermill phase3-*.ipynb phase3-*.ipynb` (in place) — that destroys the preserved Aug-2023 training outputs.

Also verified via [`tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py`](../tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py) — fast NNx-surface contract tests covering parametrized SAGE / CONV smoke-forward, `GraphAttNN(n_heads=...)` consolidation regression, and `NNParams.state()` round-trip. Runs in the CI `pytest-nnx-surface` job on every PR (`make test-nnx-surface` locally; GNN forward-pass cases skip cleanly without `pyg-lib` / `torch-sparse`).

## 5. Dependencies

- `torch` (≥ 2.0) + `torch_geometric` + `torch_sparse`
- `nnx` (PyPI: `thekaveh-nnx` — `Nets.FEED_FWD`, `Nets.GRAPH_CONV`, `Nets.GRAPH_SAGE`, `Nets.GRAPH_ATT`)
- `networkx`, `community` (python-louvain), `pandas`, `seaborn`, `matplotlib`
- `numpy`, `scikit-learn`

All available via the genai-vanilla jupyterhub image or via the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- The Reddit2 dataset (~1.5 GB) downloads into `./data/` on first run. Subsequent runs reuse the cached copy.
- GAT at hidden dim ≥ [256] hit GPU-memory ceilings on the original training hardware (M1 Max, 64GB RAM). The phase2 notebook 3 deliberately excludes GAT for that reason.
- Phase-3 notebooks are Tier-C; their preserved Aug-2023 outputs are part of the artifact and must not be re-executed in place. Verify check E5 enforces **code-cell source** equality with the `pre-cleanup-baseline` git tag (markdown and embedded outputs are not compared, so markdown edits via `scripts/edit_notebook_markdown.py` are safe).
- Memory-conscious sampling: PyG `NeighborLoader` with `[20, 15, 10]` neighborhood sizes per hop is used throughout phase 3.

## 7. Future work

- Add a GraphSAGE link-prediction variant for the planned `link_prediction-citation-graphsage-pyg` task.
- Investigate sparsified GAT (e.g., FastGAT or sparsemax attention) to overcome the memory ceiling.
- Add a `phase4-ensembling` notebook combining the SAGE and GAT predictions.
