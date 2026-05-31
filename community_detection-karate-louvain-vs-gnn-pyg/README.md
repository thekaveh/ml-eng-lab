# community_detection-karate-louvain-vs-gnn-pyg

## 1. Task summary

- **Task:** Unsupervised community detection — partition graph nodes into groups with dense within-group + sparse between-group connections.
- **Dataset:** Zachary's Karate Club via `torch_geometric.datasets.KarateClub` — 34 nodes, 78 edges, 4 true communities.
- **Models:** Louvain (`community.community_louvain.best_partition`) vs GraphSAGE encoder + KMeans on the trained embeddings.
- **Framework:** PyTorch Geometric + `python-louvain` + sklearn. Only `nnx.set_seed(0)` from nnx.

## 2. Why this exists

Two recipes for community detection:

- **Louvain** (Blondel et al., 2008) — greedy modularity-maximization, classical, fast, no hyperparameters beyond the resolution knob.
- **GraphSAGE + KMeans** — train a GNN encoder (here via link prediction, same as the sibling `link_prediction-karate-graphsage-pyg/`), then cluster the embeddings.

This notebook benchmarks both head-to-head on Karate Club, where the 4-community ground truth ships with the dataset. **Honest result:** Louvain crushes the GNN on Karate — perfect ARI/NMI = 1.000 vs the GNN's ARI 0.155 / NMI 0.429. The GNN advantage shows up on bigger graphs with richer node features; on a tiny well-modular graph with identity-only features, Louvain is the right default.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/community_detection-karate-louvain-vs-gnn-pyg/notebook.ipynb) for full rendering.

- §1 Overview — both recipes, dataset, libraries.
- §2 Environment & Setup — imports, hyperparameters (`HIDDEN_DIM=32`, `EMBED_DIM=16`, `N_EPOCHS=100`), `nnx.set_seed(0)`.
- §3 Data — `KarateClub` loader + NetworkX conversion via `to_networkx(to_undirected=True)` (Louvain wants a NetworkX graph).
- §4 Model — `GraphSAGEEncoder` definition (same as the link-prediction sibling) + Louvain / GNN contracts.
- §5 Training — Louvain (single call, no training) + GraphSAGE encoder training via link-prediction proxy (100 epochs) + KMeans on the trained embeddings.
- §6 Evaluation & Results — ARI + NMI comparison table; 3-up spring-layout visualization (truth + Louvain + GraphSAGE+KMeans labelings); §6.3 owns the "Louvain wins decisively here, GNN's loss is the *wrong proxy task* not a model defect" framing.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~6 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torch_geometric` (`KarateClub`, `SAGEConv`, `negative_sampling`, `to_networkx`).
- `networkx` — graph operations + spring-layout viz.
- `python-louvain` (the `community` module) — Louvain modularity maximization.
- `scikit-learn` — `KMeans`, `adjusted_rand_score`, `normalized_mutual_info_score`.
- `matplotlib` — spring-layout plot.
- `prettytable` — comparison table.
- `nnx` — only `nnx.set_seed(0)` for global RNG.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Louvain is "unfairly" strong on Karate.** Karate is the canonical modularity benchmark — small, well-modular, identity-only features. Louvain's perfect score isn't surprising; it's the *expected* result. The interesting comparison is on graphs where modularity isn't the right notion of similarity.
- **Link-prediction is the wrong proxy task for community detection on Karate.** GraphSAGE trained with the link-prediction BCE learns to place *connected* nodes near each other; that's not the same as placing *within-community* nodes near each other. Better proxies (GRACE / BGRL / DiffPool) explicitly push apart between-community-but-connected pairs. Out of scope here.
- **KMeans `n_clusters=4` is given.** Louvain discovers the number of communities; GraphSAGE + KMeans needs `k`. Pre-specifying `k=4` (matching ground truth) is the most charitable setup for the GNN.
- **No `nnx.NNModel.train` scaffolding** — same reasoning as the link-prediction sibling: the loop is ~15 lines and the heavier infra doesn't pay back at this scale.
- **`community` (python-louvain) is in `requirements.txt`** but optional in the verifier env — phase1 of the reddit-gnn task already imports it. This task inherits the same dep.
