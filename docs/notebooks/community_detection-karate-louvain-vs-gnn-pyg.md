# 8.15 Community detection — Karate Club Louvain vs GNN

A comprehensive walk-through of `notebooks/community_detection-karate-louvain-vs-gnn-pyg/` — the
in-repo head-to-head benchmark of classical modularity maximization (Louvain) against a
GNN-then-cluster recipe (GraphSAGE encoder + KMeans). This page is the deep-dive companion to the
task notebook: it states the problem, builds the math, dissects both recipes, reads the code top
to bottom, reports the measured results, and catalogues the pitfalls and extensions.

The notebook is **Tier-A** — CPU re-runs in roughly six seconds and it is re-executed end-to-end
in CI on every pull request. It reuses the GraphSAGE encoder from
`link_prediction-karate-graphsage-pyg/` as a feature extractor and benchmarks it against the
classical Louvain algorithm on the same Karate graph where the 4-community ground truth ships
with the dataset. The honest result — Louvain wins decisively here — is the pedagogical point.

## 8.15.1 Problem & motivation

Community detection asks: given a graph, partition its nodes into groups such that within-group
connections are dense and between-group connections are sparse. It is fully unsupervised — there
is no train/val/test split, and the ground-truth labels are used only for evaluation, never as
training signal. Two dominant recipes compete:

1. **Louvain** (Blondel et al., 2008) — greedy modularity-maximization. Iteratively moves nodes
   between communities to maximize the modularity score, a function of how much the observed
   within-community edge density exceeds the density expected under a degree-preserving null.
   Fast, no hyperparameters beyond a resolution knob, and the de-facto classical baseline.
2. **GNN + KMeans** — train a GraphSAGE encoder (here unsupervised, via a link-prediction proxy
   task identical to the sibling notebook), then cluster the trained embeddings with KMeans. The
   promise is a *richer-than-modularity* notion of similarity when node features are informative.

This notebook benchmarks both on Karate Club, where the ground-truth community structure is
well-known (the 4-way Brandes-Delling labels that ship with `torch_geometric.datasets.KarateClub`).
It exists for two reasons: to exercise `python-louvain`'s `best_partition` against a NetworkX
graph end-to-end, and to expose — honestly — the case where the GNN recipe *loses* and why. The
falsifiable hypothesis is that the GNN+KMeans recipe lands in the same ARI ballpark as Louvain;
the results section records that it does not, and the discussion owns the reason.

## 8.15.2 Concepts

| Concept | Where it shows up |
|---|---|
| Community detection | Partitioning nodes into dense-within / sparse-between groups |
| Modularity | The quality function Louvain maximizes |
| Louvain algorithm | Greedy two-phase (local move, then community aggregation) modularity maximization |
| GraphSAGE encoder | `nnx.GraphSageNN` — reused from the link-prediction sibling as a feature extractor |
| Link-prediction proxy task | The unsupervised objective used to train the GNN encoder before clustering |
| KMeans clustering | `sklearn.cluster.KMeans(n_clusters=4)` on the 16-D embeddings |
| Adjusted Rand Index (ARI) | Cluster-vs-truth agreement, corrected for chance |
| Normalized Mutual Information (NMI) | Information-theoretic cluster-vs-truth agreement |
| NetworkX interop | `torch_geometric.utils.to_networkx(to_undirected=True)` feeds Louvain |
| Reproducibility | `nnx.set_seed(0)` pins Python `random`, NumPy, PyTorch CPU + CUDA + cuDNN |

The `nnx` surface is the same thin slice as the link-prediction sibling: `set_seed`,
`Activations.RELU`, `GraphSageNN`, `NNParams`. Louvain runs through `community.community_louvain.best_partition`;
KMeans, ARI, and NMI run through scikit-learn. There is no `nnx.NNModel.train` scaffolding for
the same reason as the sibling — the loop is too short to earn its keep.

## 8.15.3 Mathematical formulation

Louvain maximizes **modularity**, the fractional excess of within-community edge density over the
density expected under a degree-preserving configuration null:

\[
Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j),
\]

where \(A\) is the (symmetric) adjacency matrix, \(k_i = \sum_j A_{ij}\) is the degree of node
\(i\), \(m = \frac{1}{2}\sum_{ij} A_{ij}\) is the number of edges, \(c_i\) is node \(i\)'s
community assignment, and \(\delta(c_i, c_j) = 1\) if \(c_i = c_j\) and \(0\) otherwise. The term
\(k_i k_j / (2m)\) is the expected number of edges between \(i\) and \(j\) if edges were rewired
at random while preserving degrees; the summand is therefore positive only when \(i\) and \(j\)
share a community *and* are connected more than chance would predict. Louvain greedily moves each
node to the neighboring community that yields the largest positive modularity gain, then
collapses each community into a super-node and repeats, until no move improves \(Q\).

The GNN side reuses the GraphSAGE neighborhood aggregation from the link-prediction sibling. For
layer \(k\), per node \(v\):

\[
h_v^{(k)} = \mathrm{ReLU}\!\left( W_{\mathrm{self}}^{(k)} h_v^{(k-1)} \;+\; W_{\mathrm{neigh}}^{(k)} \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right).
\]

The encoder maps the 34-D one-hot identity input through a 32-D hidden layer to a 16-D output
embedding \(z_v\). The encoder is trained with binary cross-entropy on positive observed edges
and freshly sampled negatives (the dot-product link-prediction proxy), exactly as in
§8.14.3 — the only difference is that the encoder here message-passes over *all* edges (there is
no link split), because the goal is a good embedding for clustering, not held-out edge prediction.

Cluster-vs-truth agreement is measured with two chance-corrected metrics. The Adjusted Rand Index
compares the pairwise co-assignment matrix to ground truth:

\[
\mathrm{ARI} = \frac{ \sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2} }{ \frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2} },
\]

where \(n_{ij}\) is the number of nodes in predicted cluster \(i\) and true class \(j\), \(a_i\)
and \(b_j\) are the row and column marginals, and \(n = 34\). ARI ranges from \(-1\) (worse than
random) through \(0\) (random) to \(1\) (perfect agreement). Normalized Mutual Information
normalizes the mutual information between the two labelings by the average of their entropies,
also landing in \([0, 1]\) with \(1\) meaning perfect agreement. Both are chance-corrected, which
is why they are preferred over plain accuracy for clustering evaluation.

## 8.15.4 Architecture

![GNN message-passing](../diagrams/img/gnn.png)

The notebook runs two independent recipes against the same graph.

**Louvain contract:**

- **Algorithm:** `community.community_louvain.best_partition(G_nx, random_state=0)`
- **Input:** NetworkX undirected graph (`to_networkx(data, to_undirected=True)`) — 34 nodes, 78 edges
- **Output:** `dict[node_id -> community_id]`; the number of communities is *discovered* by the
  algorithm, not pre-specified
- **Hyperparameters:** only `random_state=0`; the default resolution (\(1.0\)) is used

**GraphSAGE + KMeans contract:**

- **Encoder:** `GraphSageNN(NNParams(input_dim=34, hidden_dims=[32], output_dim=16, dropout_prob=0.0, activation=Activations.RELU))` — identical to the link-prediction sibling
- **Proxy loss:** `F.binary_cross_entropy_with_logits` on dot-product scores over all positive edges + freshly sampled negatives
- **Optimizer:** `torch.optim.Adam`, `lr=1e-2`, `weight_decay=5e-4`, `100` epochs (full run) or `5` (`SMOKE_TEST=1`)
- **Clustering:** `KMeans(n_clusters=4, n_init=10, random_state=0)` on the 16-D embeddings; \(k=4\) is given (matching ground truth) — the most charitable setup for the GNN

The two contracts share only the Karate graph itself. Louvain never touches the GNN; the GNN
never consults the modularity objective. The asymmetry that drives the result: Louvain's
objective *is* the within-community density signal, whereas the GNN's proxy objective (link
prediction) optimizes a related but distinct quantity — whether connected pairs score highly —
which does not explicitly separate within-community from between-community connectivity.

## 8.15.5 Code walkthrough

### Graph load and NetworkX conversion

```python
karate = KarateClub()
data = karate[0]
y_true = data.y.cpu().numpy()

G_nx = to_networkx(data, to_undirected=True)
```

`data.y` carries the 4-way Brandes-Delling ground-truth labels that PyG ships with the dataset;
they are held back for evaluation only. `to_networkx(..., to_undirected=True)` collapses the
doubled directed edge storage (156 entries) into the 78 undirected edges that
`python-louvain` expects.

### Louvain (single call, no training)

```python
partition_louvain = community_louvain.best_partition(G_nx, random_state=0)
pred_louvain = [partition_louvain[i] for i in range(data.num_nodes)]
n_communities_louvain = len(set(pred_louvain))
```

`best_partition` returns a node-to-community dict. The number of communities is discovered by the
algorithm — Louvain does not need \(k\) handed to it, which is one of its operational advantages
over KMeans.

### GraphSAGE encoder training (link-prediction proxy)

```python
encoder = GraphSageNN(
    NNParams(input_dim=data.num_features, hidden_dims=[HIDDEN_DIM],
             output_dim=EMBED_DIM, dropout_prob=0.0, activation=Activations.RELU)
).to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

for epoch in range(N_EPOCHS):
    encoder.train()
    optimizer.zero_grad()
    z = encoder(data.x.to(DEVICE), data.edge_index.to(DEVICE))
    pos_ei = data.edge_index.to(DEVICE)
    neg_ei = negative_sampling(edge_index=pos_ei, num_nodes=data.num_nodes,
                               num_neg_samples=pos_ei.size(1))
    ei = torch.cat([pos_ei, neg_ei], dim=1)
    label = torch.cat([torch.ones(pos_ei.size(1)), torch.zeros(neg_ei.size(1))]).to(DEVICE)
    loss = F.binary_cross_entropy_with_logits(decode(z, ei), label)
    loss.backward()
    optimizer.step()
```

This is the same encoder + dot-product + BCE recipe as §8.14.5, with two differences: there is no
`RandomLinkSplit` (the encoder trains over *all* observed edges, since the goal is a good
embedding for clustering, not held-out edge prediction), and the positive set is the full
`data.edge_index` rather than a train-split `edge_label_index`.

### KMeans on the trained embeddings

```python
encoder.eval()
with torch.no_grad():
    embeddings = encoder(data.x.to(DEVICE), data.edge_index.to(DEVICE)).cpu().numpy()

km = KMeans(n_clusters=len(set(y_true)), n_init=10, random_state=0).fit(embeddings)
pred_gnn = km.predict(embeddings)
```

`n_clusters=4` is set from the ground-truth count — the most charitable setup for the GNN, since
Louvain has to discover \(k\) on its own. `n_init=10` reruns KMeans from ten random inits and
keeps the best, which removes one source of clustering noise.

### Agreement metrics

```python
ari_louvain = adjusted_rand_score(y_true, pred_louvain)
nmi_louvain = normalized_mutual_info_score(y_true, pred_louvain)
ari_gnn     = adjusted_rand_score(y_true, pred_gnn)
nmi_gnn     = normalized_mutual_info_score(y_true, pred_gnn)
```

Both metrics are chance-corrected and invariant to label permutation (the predicted cluster ids
have no inherent ordering), which is why they are the right yardstick for clustering quality.

## 8.15.6 Results & analysis

On the seeded (`nnx.set_seed(0)`, `random_state=0`) run, the two recipes land as:

| Recipe | n communities | ARI | NMI |
|---|---|---|---|
| Louvain (modularity) | 4 | 1.000 | 1.000 |
| GraphSAGE + KMeans | 4 | 0.155 | 0.429 |

The GraphSAGE encoder's training BCE dropped from 0.7064 to 0.3625 over 100 epochs — the proxy
task converged, so the loss is not the failure mode. Three observations:

1. **Louvain recovers the ground truth exactly.** ARI = NMI = 1.000 means every node lands in its
   true community. This is the *expected* result: Karate is the canonical modularity benchmark,
   small and strongly modular, and the 1977 administrator-vs-trainer fracture that made the
   dataset famous is precisely the structure modularity maximization is designed to find.
2. **GraphSAGE + KMeans lands far behind (ARI 0.155, NMI 0.429).** The encoder trained cleanly on
   the link-prediction proxy, but the resulting embeddings do not separate the four communities
   the way modularity does. The proxy converged; the embeddings just do not encode community
   structure as their dominant axis of variation.
3. **The gap is the *wrong proxy task*, not a model defect.** Link-prediction BCE trains the
   encoder to place *connected* nodes near each other in embedding space. That is the right
   objective for predicting edges, but it does not distinguish *within-community* connected pairs
   from *between-community-but-connected* bridge pairs — and bridges are exactly the edges that
   determine community boundaries. Modularity, by construction, down-weights bridges; the
   dot-product BCE does not.

The pedagogical headline is that picking the right unsupervised proxy for a downstream
evaluation is a real modeling decision. The GNN recipe wins on bigger, less-modular graphs with
rich node features (citation networks, biological networks, social graphs with user attributes)
where modularity is too coarse a similarity notion — but on Karate, Louvain is the right default.

## 8.15.7 Pitfalls & edge cases

- **Karate is "unfairly" strong for Louvain.** It is the poster-child dataset for
  modularity-maximization: small, well-modular, identity-feature-only. Louvain's perfect score
  is the expected result, not a surprise. Do not generalize the Louvain-wins headline to graphs
  where modularity is not the right notion of similarity.
- **Link prediction is the wrong proxy for community detection on Karate.** GraphSAGE trained
  with link-prediction BCE learns to place connected nodes near each other; that is not the same
  as placing within-community nodes near each other. Better proxies (GRACE, BGRL, DiffPool)
  explicitly push apart between-community-but-connected pairs. The notebook deliberately does
  not switch proxies — the point is to expose the mismatch.
- **KMeans needs \(k\); Louvain does not.** Pre-specifying `n_clusters=4` (matching ground truth)
  is the most charitable setup for the GNN — Louvain has to discover the count on its own. A
  less charitable setup (e.g. selecting \(k\) by silhouette) would widen the gap further.
- **Identity features cap the GNN's achievable score.** Karate has no real node attributes; the
  34-D one-hot input gives the encoder only connectivity to work with. On graphs with real node
  features the GNN+KMeans recipe can integrate feature similarity and connectivity jointly,
  which is where it starts to outperform modularity-only methods.
- **No `nnx.NNModel.train` scaffolding, deliberately.** Same reasoning as the link-prediction
  sibling: the loop is short enough that the heavier checkpoint / scheduler / callback
  infrastructure does not pay back at this scale.
- **`python-louvain` (`community`) is an unconditional dep.** It is already imported by the
  reddit-gnn task's phase1, so this task adds no new pin to `requirements.txt`.
- **Read ARI and NMI together, not in isolation.** ARI is the stricter metric (it penalizes
  chance agreement symmetrically); NMI can be inflated by a large number of small clusters. The
  notebook reports both, and on Karate they tell the same story.

## 8.15.8 Extensions & references

- **Swap in a contrastive proxy that separates communities.** GRACE (Zhu et al., 2020), BGRL
  (Thakoor et al., 2021), or DiffPool (Ying et al., 2018) explicitly push apart
  between-community node pairs during encoder training; retraining the GraphSAGE encoder with
  one of these objectives and re-running KMeans is the natural follow-up that tests whether the
  GNN recipe can close the gap on Karate.
- **Move to a bigger, less-modular graph.** Louvain's perfect score here is partly a property of
  Karate. On citation networks (Cora, Citeseer), biological networks (PPI), or social graphs
  with user attributes, the GNN+KMeans recipe can win because it integrates multi-hop structure
  and node features jointly. The encoder + KMeans plumbing ports over unchanged.
- **Reuse the link-prediction encoder from the sibling notebook.** The GraphSAGE encoder trained
  here is the same model as in `link_prediction-karate-graphsage-pyg.md` — read that page for the
  full encoder + decoder + split walkthrough. The two notebooks share the encode-then-score
  contract; what differs is the downstream consumer (KMeans here, the AUC metric there).
- **Discover \(k\) instead of pre-specifying it.** Replace the fixed `n_clusters=4` with a
  silhouette-score or modularity-based \(k\) selector; this is the fairer comparison against
  Louvain, which discovers the count automatically.
- **Blondel, V. D. et al. (2008).** "Fast unfolding of communities in large networks." The
  original Louvain paper. Brandes, U. et al. (2008), "On modularity clustering," is the
  canonical reference for the modularity quality function and its resolution limit.
