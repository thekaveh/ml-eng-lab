# text_classification-agnews-spacy-mlp-pytorch

## 1. Task summary

- **Task:** 4-class topic classification (World / Sports / Business / Sci-Tech) — AG-News style.
- **Dataset:** Embedded ~80 hand-written news-headline corpus (20/class), tiled 4× for training-set size + vocabulary coverage. AG-News-style 4-topic split; **not** the full AG-News (which would require a network download).
- **Model:** spaCy tokenizer + bag-of-words featurizer + `nnx.NNModel` MLP (`vocab_size → 64 → 4`).
- **Framework:** PyTorch (via [`nnx`](../nnx)) + spaCy + sklearn.

## 2. Why this exists

Text classification is the canonical pre-transformer NLP task. The dominant pipeline shape — **tokenize → vectorize → classify** — predates Transformers and is still the right baseline for many production text classifiers. This notebook lands that recipe in the collection with the simplest vectorizer (BoW) and the smallest plausible model (1-hidden-layer MLP).

This is the first **NLP classification** task in ml-lab (the existing text-related task `text_generation-tinyshakespeare-transformer-pytorch/` is autoregressive generation, not classification). It pairs naturally with the future `text_classification-imdb-distilbert-hf` roadmap entry (transformer-based) — same task family, different vectorizer + bigger dataset.

The corpus is embedded inline (~80 hand-written headlines) so the notebook is fully self-contained — no `torchtext.datasets.AG_NEWS` download, no HuggingFace dataset auth, no issue #3-style network hang in CI.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/text_classification-agnews-spacy-mlp-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — text-classification recipe, dataset framing, libraries; positioning vs `text_generation-tinyshakespeare-transformer-pytorch/`.
- §2 Environment & Setup — imports (spaCy `en_core_web_sm`), hyperparameters (`VOCAB_SIZE=200`, `HIDDEN_DIMS=[64]`, `N_EPOCHS=80`, `CORPUS_REPEAT=4`), `nnx.set_seed(0)`.
- §3 Data — embedded 4-topic corpus (20 headlines/class, tiled 4×), spaCy lemmatize + lowercase + drop stopwords/punctuation, train/test split, **train-only vocabulary** (the standard footgun: leaking test tokens into the vocab inflates accuracy), L2-normalized bag-of-words featurizer.
- §4 Model — small `FeedFwdNN` MLP + positioning vs future transformer text-classification tasks.
- §5 Training — `nnx.NNModel.train` with `test_loader` as `val_loader` (tiny corpus — used for visibility, not for early-stopping).
- §6 Evaluation & Results — test accuracy + per-class precision / recall / f1 via sklearn `classification_report`; confusion matrix heatmap; §6.3 scaling levers (TF-IDF, n-grams, pretrained embeddings, transformer fine-tuning).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier A** (cheap, ~13 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (the submodule) — `FeedFwdNN`, `NNModel`, `NNTrainParams`, `set_seed`.
- `spacy` — tokenizer + lemmatizer. **Note**: `en_core_web_sm` (the small English model) must be installed separately via `python -m spacy download en_core_web_sm`. CI installs it explicitly (not via `requirements.txt`).
- `scikit-learn` — train/test split, accuracy + classification_report + confusion_matrix.
- `numpy` — feature matrix.
- `matplotlib` — confusion-matrix heatmap.

`spacy` and `python-louvain` are already in the root `requirements.txt`; the `en_core_web_sm` model download is the only extra step.

## 6. Known issues

- **Embedded corpus is tiny.** 80 unique headlines × 4 tiling = 320 documents. Real AG-News is 120k. Absolute accuracy numbers are not directly comparable.
- **Vocab is train-only** — the standard recipe. Test-time tokens not in vocab are simply ignored (zero featurization weight). This is correct behavior; it would matter more at larger vocab sizes.
- **`en_core_web_sm` is a separate install.** `pip install spacy` doesn't pull the English model. CI step: `python -m spacy download en_core_web_sm`. Document this on every contributor onboarding.
- **No baseline.** A standard pipeline would compare against `sklearn.naive_bayes.MultinomialNB` or `sklearn.linear_model.LogisticRegression` on the same BoW features; the MLP doesn't necessarily win at this corpus scale. Adding sklearn baselines is queued.
- **No bigram / TF-IDF features.** Both would help a real run; this notebook stays at the simplest baseline so the *recipe shape* is unambiguous.
