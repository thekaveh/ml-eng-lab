# 8.17 Text classification — AG-News spaCy + MLP

A comprehensive walk-through of `notebooks/text_classification-agnews-spacy-mlp-pytorch/` — the
first in-repo NLP *classification* task and the canonical exemplar of the
**tokenize → vectorize → classify** recipe that predates Transformers and still underpins most
production text classifiers. This page is the deep-dive companion to the task notebook: it states
the problem, builds the math, dissects the pipeline, reads the code top to bottom, reports the
measured accuracy, and catalogues the pitfalls — chief among them the **train-only vocabulary**
footgun that silently inflates evaluation accuracy when handled wrong.

The notebook is **Tier-A** — CPU re-runs in about thirteen seconds and it is re-executed end-to-end
in CI on every pull request. The corpus is embedded inline (80 hand-written headlines, 20 per
topic, tiled 4×) so the notebook is fully self-contained: no `torchtext.datasets.AG_NEWS` download,
no HuggingFace auth, no network hang in CI. It pairs naturally with the future
`text_classification-imdb-distilbert-hf` roadmap entry — same task family, transformer vectorizer
instead of BoW.

## 8.17.1 Problem & motivation

AG-News is the canonical four-topic news-headline classification benchmark (World / Sports /
Business / Sci-Tech). This notebook solves an AG-News-*style* version of it on an embedded
hand-written corpus, deliberately small so the entire pipeline — tokenization, vocabulary
construction, featurization, training, evaluation — fits in one screen and runs in seconds on a
laptop CPU.

The notebook exists for three reasons:

1. **Land the pre-transformer NLP recipe.** The dominant text-classification pipeline shape —
   tokenize → vectorize → classify — predates Transformers and is still the right baseline for
   many production text classifiers (high-throughput routing, spam, topic tagging). This notebook
   lands that recipe with the simplest vectorizer (bag-of-words) and the smallest plausible model
   (one-hidden-layer MLP) so the *recipe shape* is unambiguous.
2. **First NLP classification task in the lab.** The existing text task
   (`text_generation-tinyshakespeare-transformer-pytorch`, §8.16) is autoregressive generation, not
   classification. This notebook is the classification counterpart and exercises the
   `FeedFwdNN` + `NNModel` core on text features the same way §8.1 exercised it on tabular
   features.
3. **Make the train-only-vocabulary footgun explicit.** The standard silent error in
   text-classification pipelines is building the vocabulary over the *entire* corpus (train +
   test), leaking test-only tokens into the featurizer and inflating accuracy. The notebook builds
   the vocabulary from the train split *only* and flags the choice in a code comment so a reader
   copying the recipe cannot miss it.

The falsifiable hypothesis tested by the notebook is that a single hidden layer over a 200-dim L2-
normalized BoW featurization clears ~80% test accuracy on this four-topic corpus — i.e. that the
top-200 lemma vocabulary is discriminative enough to separate the topics even with one-hot order
destroyed.

## 8.17.2 Concepts

| Concept | Where it shows up |
|---|---|
| Multi-class text classification | Four AG-News topics; one correct label per headline |
| spaCy lemmatization + stopword/punct removal | `nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])` then lemma filter |
| Bag-of-words (BoW) featurization | Counts of vocab-token lemmas per document — order discarded |
| L2-normalized counts | Each BoW vector divided by its L2 norm so doc length doesn't dominate the dot product |
| Train-only vocabulary | `Counter` built from train docs *only*; `most_common(200)` keeps the top-K |
| MLP classifier | `nnx.FeedFwdNN` with `hidden_dims=[64]`, ReLU, dropout 0 |
| Softmax + cross-entropy | The training objective over the four logits |
| Macro-averaged precision/recall/f1 | Per-class metrics averaged equally — right under the balanced 16-per-class test split |
| Stratified train/test split | `train_test_split(..., stratify=labels_np)` keeps all four topics balanced |

The `nnx` surface consumed: `NNModel`, `NNParams`, `NNModelParams`, `NNTrainParams`,
`NNOptimParams`, plus the enums `Activations`, `Devices`, `Losses`, `Nets`, `Optims`, `VisUtils`,
and `nnx.set_seed`. The text-specific half of the pipeline (spaCy, sklearn, numpy) sits outside
`nnx` and is glued together by hand — this is the *pre-transformer* recipe, before any of it was
absorbed into a unified API.

## 8.17.3 Mathematical formulation

The classifier maps a document to one of four topic logits. The pipeline has three stages, each
with its own math.

**Tokenization.** Each headline string is mapped to a sequence of lowercased lemmas with stopwords
and punctuation removed. This is a deterministic filter, not a continuous transform; it reduces the
vocabulary to its discriminative content words.

**Bag-of-words featurization.** Let \(V\) be the train-only vocabulary (the \(K = 200\) most
frequent train lemmas). For a document with token-multiset \(T\), the raw BoW vector is

\[
\tilde{x}_v = \mathrm{count}(v \text{ in } T), \qquad v = 1, \dots, K.
\]

Tokens not in the vocabulary (out-of-vocabulary at test time, or below the frequency cutoff) are
silently dropped — they contribute zero. The vector is then L2-normalized:

\[
x = \frac{\tilde{x}}{\|\tilde{x}\|_2}, \qquad \|\tilde{x}\|_2 = \sqrt{\sum_v \tilde{x}_v^2}.
\]

L2 normalization removes the effect of document length so that a 12-token headline and a 6-token
headline with the same *relative* lemma distribution map to the same feature vector. (TF-IDF would
further down-weight corpus-frequent tokens; this notebook stays at the simplest baseline.)

**Classifier.** The MLP produces a four-vector of logits \(z = (z_0, z_1, z_2, z_3)\) via a single
hidden layer with ReLU:

\[
z = W_2 \, \mathrm{ReLU}(W_1 x + b_1) + b_2,
\]

with shapes \(W_1 \in \mathbb{R}^{64 \times 200}\), \(W_2 \in \mathbb{R}^{4 \times 64}\). The
softmax + cross-entropy objective is identical to the Iris tabular case (§8.1.3):

\[
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
\mathcal{L}(z, c) = -\log \hat{y}_{c},
\]

where \(c\) is the index of the correct topic. The optimizer is Adam with \(\eta = 5 \times 10^{-3}\),
weight decay \(10^{-3}\), and momentum \((0.9, 0.999)\).

## 8.17.4 Architecture

![Feed-forward MLP](../diagrams/img/mlp.png)

The pipeline has two stages — a frozen hand-written featurizer and a learned MLP head.

**Featurizer (frozen):** spaCy lemmatizer + 200-dim train-only BoW + L2 normalization. No learned
parameters; the entire "representation learning" is the choice of vocabulary and the L2 norm.

**Classifier (`Nets.FEED_FWD`):** a single-hidden-layer MLP, identical in shape to the Iris
Candidate B:

| Layer | Width | Activation |
|---|---|---|
| Input | 200 (vocab size) | — |
| Hidden | 64 | ReLU |
| Output | 4 (topics) | softmax (via cross-entropy) |

`dropout_prob=0.0` — with 256 train docs and a 64-unit hidden layer, heavier dropout starves the
head of signal; the L2 normalization and weight decay already regularize.

The shared contract — everything held constant:

- **Net:** `Nets.FEED_FWD`
- **Loss:** `Losses.CROSS_ENTROPY`
- **Optimizer:** `Optims.ADAM`, `max_lr=5e-3`, `momentum=(0.9, 0.999)`, `weight_decay=1e-3`
- **Device:** `Devices.CPU`
- **Epochs:** `80` (full run) or `5` (`SMOKE_TEST=1` for CI)
- **Batch size:** `16`
- **Seed:** `0`

The data: 80 unique headlines (20 per topic) tiled `CORPUS_REPEAT=4` → 320 documents, split 80/20
stratified → 256 train / 64 test (16 per topic in test). The train docs surface 410 unique lemmas;
the top 200 by train frequency are kept as the vocabulary. The test loader is wired as the
`val_loader` for visibility into per-epoch test loss — a deliberate choice on a tiny corpus, flagged
in the pitfalls.

## 8.17.5 Code walkthrough

### spaCy lemmatization

```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def tokenize(text):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc
            if not t.is_stop and not t.is_punct and t.lemma_.strip()]
```

The parser and NER pipes are disabled for speed — only the tokenizer + tagger (needed for
lemmatization) run. The filter keeps content-word lemmas: lowercased, non-stopword, non-punct,
non-empty-after-strip. The `t.lemma_.strip()` guard also drops whitespace/SPACE tokens (whose
lemma is the whitespace itself).

### The train-only vocabulary (the footgun)

```python
counter = Counter()
for i in train_idx:                          # <- train split ONLY
    counter.update(tokens_per_doc[i])
vocab = {tok: idx for idx, (tok, _) in enumerate(counter.most_common(VOCAB_SIZE))}
# Vocab from TRAIN ONLY — leaking test tokens into the vocab is a real evaluation
# footgun in text-classification pipelines.
```

This is the single most important line in the notebook. `for i in train_idx` (not `range(len(texts))`)
guarantees that test-only tokens never enter the vocabulary. If the loop ran over all docs, the
featurizer would "see" the test distribution at vocabulary-construction time — a subtle form of
data leakage that inflates test accuracy by a few points and is invisible unless you audit the
vocab-construction loop. Recorded: 410 unique lemmas in train, top 200 kept. The vocab maps
token → dense index ordered by descending train frequency (so index 0 is the most frequent lemma).

### L2-normalized BoW featurizer

```python
def featurize(toks):
    v = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for tok in toks:
        if tok in vocab:                      # OOV tokens silently dropped
            v[vocab[tok]] += 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v /= n                                # L2-normalize so doc length doesn't dominate
    return v
```

Out-of-vocabulary tokens (test-only lemmas, or below-top-200 lemmas) hit the `if tok in vocab`
guard and contribute zero — this is the *correct* behavior and the flip side of the train-only
vocab choice: at test time the model must work with the vocabulary it learned, exactly as it would
in production. The L2 normalization projects every BoW vector onto the unit sphere so that the
downstream dot-product logits measure *direction*, not magnitude.

### MLP head and training

```python
def make_model():
    return NNModel(
        net_params=NNParams(input_dim=VOCAB_SIZE, output_dim=len(TOPIC_NAMES),
                            hidden_dims=HIDDEN_DIMS, dropout_prob=0.0,
                            activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=DEVICE, loss=Losses.CROSS_ENTROPY),
    )

run = model.train(
    params=NNTrainParams(
        n_epochs=N_EPOCHS, train_loader=train_loader,
        val_loader=test_loader,               # tiny corpus — visibility, not early-stopping
        optim=NNOptimParams(name=Optims.ADAM, max_lr=LR,
                            momentum=(0.9, 0.999), weight_decay=1e-3),
    ),
)
```

The `val_loader=test_loader` wiring is a deliberate small-corpus choice: with only 64 test docs
there is no separate val split to carve off, so the test loss is surfaced per-epoch for visibility
into convergence — but it is *not* used for early-stopping or model selection. Training runs 16
batches/epoch × 80 epochs = **1280 iterations**.

### Evaluation

```python
y_pred = np.argmax(model.predict_proba(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=TOPIC_NAMES))
VisUtils.confusion_matrix(Y_true=y_test, Y_pred=y_pred, class_names=TOPIC_NAMES, ...)
```

`classification_report` surfaces per-class precision/recall/f1; the confusion-matrix heatmap (via
`VisUtils`) is the directly-interpretable artifact showing which topic pairs get confused.

## 8.17.6 Results & analysis

On the recorded Tier-A run (`SMOKE_TEST=0`, 80 epochs, 1280 iterations) the loss trajectory is
train loss `1.3738 → 0.1120`, val loss `1.3172 → 0.2770` across the 80 epoch-end probes. Test
accuracy **85.94%** (55/64 correct). Per-class metrics, recorded verbatim:

| Topic | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| world | 1.00 | 0.81 | 0.90 | 16 |
| sports | 0.87 | 0.81 | 0.84 | 16 |
| business | 0.70 | 1.00 | 0.82 | 16 |
| sci-tech | 1.00 | 0.81 | 0.90 | 16 |
| **macro avg** | **0.89** | **0.86** | **0.86** | 64 |

Three observations:

1. **Business is over-predicted.** It has recall 1.00 (every business test doc is caught) but
   precision 0.70 — meaning several non-business docs get routed into business. The confusion
   matrix shows business absorbing overflow from sci-tech and world. This is the typical failure
   mode when one topic's vocabulary overlaps another's (business ↔ sci-tech is the classic AG-News
   overlap; world ↔ sports is the other).
2. **World and sci-tech have precision 1.00.** When the model predicts them it is always right —
   the business over-prediction is what costs the remaining points. A TF-IDF featurizer or a
   bigram vocabulary would likely tighten the business boundary.
3. **The 86% headline is on a 64-doc test split.** Each misclassification costs ~1.5 percentage
   points. The notebook pins `random_state=0` so the result is reproducible, but a different seed
   would move the boundary by a point or two. Read the headline as "on this split," not "on
   AG-News" — real AG-News at 120 k docs lands near 90–91% with this exact recipe.

## 8.17.7 Pitfalls & edge cases

- **Train-only vocabulary is the load-bearing choice.** The single most common silent error in
  text-classification pipelines is building the vocabulary over the *entire* corpus. The
  featurizer then "knows" about test-only tokens at vocabulary-construction time — a subtle leak
  that inflates accuracy by a few points. The notebook loops `for i in train_idx` and flags the
  choice in a code comment. If you copy this recipe, audit this loop first.
- **`en_core_web_sm` is a separate install.** `pip install spacy` does not pull the English model.
  CI runs `python -m spacy download en_core_web_sm` as a dedicated step
  (`.github/workflows/ci.yml`, `tier-a-papermill` job); local contributors must run it once after
  `pip install -r requirements.txt` or `spacy.load(...)` raises `OSError`. The notebook has no
  in-notebook download fallback — it fails loudly if the model is missing.
- **`val_loader=test_loader` is visibility, not model selection.** On a 64-doc test split there is
  no val set to carve off, so the test loss is surfaced per-epoch for convergence visibility but
  is *not* used for early-stopping. This is defensible at this scale; it is *not* the pattern to
  copy for a real 120 k-doc run, where you would hold out a proper val split for hyperparameter
  selection.
- **No baseline.** A standard pipeline compares the MLP against
  `sklearn.naive_bayes.MultinomialNB` or `sklearn.linear_model.LogisticRegression` on the same BoW
  features. At this corpus scale the MLP does not necessarily win; the notebook ships without that
  comparison (queued) and the §8.17.6 numbers should be read as "this MLP's score," not "BoW-MLP's
  ceiling."
- **BoW destroys word order.** "Stock market closes at record high" and "high record at closes
  market stock" featurize identically. Negation, scope, and phrase-level meaning are all lost. For
  topic classification this is mostly fine (topics are keyword-driven); for sentiment it is a much
  bigger loss — see §8.18.
- **The embedded corpus is tiny.** 80 unique headlines × 4 tiling = 320 documents. Real AG-News is
  120 k. Absolute accuracy is not directly comparable; the *recipe shape* and the pitfalls are the
  transferable lessons.

## 8.17.8 Extensions & references

- **Add sklearn baselines on the same BoW features.** `MultinomialNB` and `LogisticRegression`
  are the standard BoW baselines; running both on the same featurizer isolates what the MLP's
  hidden layer actually buys (likely nothing at this scale). This is the queued comparison the
  notebook ships without.
- **Upgrade the featurizer to TF-IDF + bigrams.** Term-frequency–inverse-document-frequency
  down-weights corpus-frequent tokens (the/and/of) that the stopword filter misses; bigrams
  (`ngram_range=(1,2)`) recover minimal phrase structure ("stock market", "interest rate"). Both
  typically add 2–5 accuracy points on AG-News.
- **Swap BoW for pretrained embeddings.** Average GloVe or fastText vectors per document as the
  feature vector instead of BoW counts. This captures word similarity (market ≈ economy) that
  one-hot BoW cannot, and is the natural stepping stone toward a transformer fine-tune.
- **Scale to real AG-News via `torchtext.datasets.AG_NEWS` or HuggingFace.** 120 k training docs
  land this recipe near 90–91% accuracy; the bottleneck moves from featurization to train-time
  throughput. Requires solving the issue #3 network-download-in-CI problem first (cache the
  dataset, or fetch in a setup step outside the papermill run).
- **Reference reading.** Manning, Raghavan & Schütze, *Introduction to Information Retrieval*
  (2008), ch. 13–14 for BoW + TF-IDF + Naive Bayes; Jurafsky & Martin, *SLP* (3rd ed.), ch. 4–5
  for logistic regression on text. The spaCy lemmatizer docs (`spacy.io/api/lemmatizer`) cover
  the `lemma_.lower()` filter pattern this notebook uses.
