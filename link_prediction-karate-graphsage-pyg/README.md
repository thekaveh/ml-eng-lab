# link_prediction-karate-graphsage-pyg

## 1. Task summary

- **Task:** Link prediction — given a partially-observed graph, predict which missing edges are most likely to actually exist.
- **Dataset:** Zachary's Karate Club (1977) via `torch_geometric.datasets.KarateClub` — 34 nodes (members), 78 undirected friendship edges, 34-D one-hot identity features.
- **Model:** 2-layer GraphSAGE encoder (`SAGEConv` 34→32→16) + dot-product edge scorer (`σ(z_u · z_v)`).
- **Framework:** PyTorch + PyTorch Geometric. **Minimal `nnx`** — only `nnx.set_seed(0)` for global RNG; the link-prediction loop is small enough that the `nnx.NNModel.train` scaffolding doesn't pay back.

## 2. Why this exists

Link prediction is the canonical GNN evaluation task for unsupervised graph representation learning. The recipe: encode each node to a vector via a GNN, score each edge as the dot-product of its endpoints' embeddings, train with BCE on (positive observed edges, negative-sampled non-edges), evaluate AUC + Average Precision on held-out edges.

This is a smaller, focused complement to `node_classification-reddit-gnn-pyg/`. That task uses GraphSAGE for *supervised node classification* on a huge graph (Reddit2, 232k nodes). This one uses GraphSAGE as an *unsupervised edge encoder* on a tiny benchmark graph. Same architecture, different task signal (the graph's own edges vs externally-provided labels).

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/link_prediction-karate-graphsage-pyg/notebook.ipynb) for full rendering.

- §1 Overview — link prediction recipe, dataset, libraries; framed against the reddit-gnn sibling.
- §2 Environment & Setup — imports, hyperparameters (`HIDDEN_DIM=32`, `EMBED_DIM=16`, `N_EPOCHS=200`), `nnx.set_seed(0)` (the only nnx usage — just for global seeding).
- §3 Data — `KarateClub` loader + `RandomLinkSplit(num_val=0.1, num_test=0.2)`; explains the `train_data.edge_index` (message-passing) vs `train_data.edge_label_index` (supervised positive edges) distinction.
- §4 Model — 2-layer `SAGEConv` encoder + dot-product `decode(z, edge_index)`.
- §5 Training — re-sample fresh negatives per epoch via `negative_sampling`; BCE loss; track train loss + val AUC.
- §6 Evaluation & Results — test AUC + AP table + training-trajectory + val-AUC curves; §6.3 explains the small-test-set variance.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~5 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torch_geometric` (`KarateClub`, `SAGEConv`, `RandomLinkSplit`, `negative_sampling`).
- `scikit-learn` — `roc_auc_score`, `average_precision_score`.
- `matplotlib` — training trajectory + val AUC curve.
- `prettytable` — final metrics table.
- `nnx` — only `nnx.set_seed(0)` (global RNG seeding).

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Test AUC is noisy.** The test set is only 30 edges (15 pos + 15 neg). One or two "structural-bridge" edges between communities can swing AUC by ~10 pp run-to-run. The recorded run has val AUC ~0.74 but test AUC ~0.43 — that gap is *the test set being too small*, not the model being broken. The §6.3 prose calls this out and points at bigger graphs (Reddit2, Cora, OGBL benchmarks) where the metrics stabilize.
- **Identity features.** Karate has no real node attributes; the input `x` is a 34-D one-hot identity matrix. GraphSAGE has to derive embeddings purely from connectivity. On graphs with real node features (text, biological annotations, user profiles), the same recipe usually performs much better.
- **No `nnx.NNModel.train` scaffolding.** The training loop is ~15 lines; `nnx`'s checkpoint / scheduler / callback infra doesn't pay back at this scale. Heavier link-prediction notebooks (e.g. the future `link_prediction-citation-graphsage-pyg` from the README roadmap) would benefit from `nnx` wrapping.
- **`add_negative_train_samples=False`.** We resample fresh negatives each epoch (the recommended pattern). The alternative (sample once at split time) speeds the epoch up but invites memorization of a fixed negative set.
- **Single seed.** Recorded numbers depend on `RandomLinkSplit`'s seed and `negative_sampling`'s draws. Average across seeds for a robust estimate; we don't here to keep the notebook simple.
