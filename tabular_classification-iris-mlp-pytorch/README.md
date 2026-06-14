# tabular_classification-iris-mlp-pytorch

## 1. Task summary

- **Task:** Multi-class tabular classification.
- **Dataset:** Iris (150 samples × 4 features × 3 classes; `sklearn.datasets.load_iris`).
- **Model:** Feed-forward neural network — using `nnx.FeedFwdNN`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)).

## 2. Why this exists

The first in-repo use of `nnx.NNTabularDataset`, the pandas-DataFrame → `DataLoader` wrapper introduced in the 21-commit `nnx` hop. Iris is small enough that a CPU re-run completes in seconds, which makes it the right vehicle for teaching the tabular-classification pattern without dataset noise distracting from the API surface.

The source lesson (CS5644 `MLP-updated.ipynb`) used sklearn's blackbox `MLPClassifier` with KFold. This re-telling switches to `nnx.NNModel` so the full training loop is visible — per-epoch loss curves via `VisUtils.multi_line_plot`, validation tracking, confusion matrices via `VisUtils.confusion_matrix`. The same `FeedFwdNN` + `NNModel` core that drives `image_classification-mnist-ffnn-pytorch`. Run-to-run determinism is pinned via `nnx.set_seed(0)`.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/tabular_classification-iris-mlp-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — task, dataset, model family, headline-result preview.
- §2 Environment & Setup — `nnx.set_seed(0)`, version stamps, imports.
- §3 Data — iris schema, summary stats, class balance, pairplot, MinMax scaling, 70/15/15 train/val/test split, post-cleaning recap.
- §4 Model — three candidate FFN topologies (`hidden_dims=[]` linear baseline, `[8]` one hidden, `[16, 8]` two hidden) with *a priori* reasoning for each.
- §5 Training — all three candidates trained side-by-side via `NNModel.train`; per-candidate `NNRun` captured; train/val loss overlaid via `VisUtils.multi_line_plot`.
- §6 Evaluation & Results — comparison table (precision / recall / f1 / accuracy on the held-out test split), side-by-side confusion matrices via `VisUtils.confusion_matrix`, per-candidate per-class diagnostic plot, and the headline "which candidate won and why" verdict.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open notebook.ipynb in VS Code (attached to the running container) or browser.
# Run all cells.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, <2 s of compute on CPU; <30 s wall-clock including loader setup). Re-executed in CI on every PR.

Also verified via [`tests/nnx_surface/test_tabular_classification_iris_mlp_pytorch.py`](../tests/nnx_surface/test_tabular_classification_iris_mlp_pytorch.py) — a fast NNx-surface contract test pinning the `NNTabularDataset` → `FeedFwdNN` → `NNModel` call chain. Runs in the CI `pytest-nnx-surface` job on every PR (`make test-nnx-surface` locally).

## 5. Dependencies

- `nnx` (PyPI: `thekaveh-nnx`)
- `torch` (≥ 2.0)
- `pandas`, `numpy`, `scikit-learn`
- `seaborn`, `matplotlib`

All installed by the genai-vanilla jupyterhub image or via the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- No persisted data download — iris ships inside scikit-learn. `./data/` is therefore unused (still gitignored at repo level for consistency).
- `./runs/` is gitignored; first run creates a fresh runs directory.

## 7. Future work

- Add a `tabular_classification-titanic-mlp-pytorch` sibling that scales the same pattern to 891 samples + categorical encoding — informs the planned `tabular_classification-titanic-xgboost-sklearn` task.
- Replace the linear-baseline candidate with `sklearn.LogisticRegression` for an apples-to-apples "learned-loop vs closed-form" comparison.
