# 8.18 Sentiment classification — VADER vs MLP

A comprehensive walk-through of `notebooks/sentiment_classification-vader-mlp-pytorch/` — the
in-repo head-to-head between a hand-tuned rule-based lexicon (VADER) and a learned neural
classifier (spaCy BoW → MLP) on the same embedded sentiment corpus. This page is the deep-dive
companion to the task notebook: it states the problem, builds the math for both recipes, dissects
each pipeline, reads the code top to bottom, reports the measured head-to-head accuracy, and lands
the production-relevant lesson — **hand-tuned lexicons are much harder to beat than the supervised-
ML literature acknowledges, so always include VADER as a baseline.**

The notebook is **Tier-A** — CPU re-runs in about ten seconds and it is re-executed end-to-end in
CI on every pull request. The corpus is embedded inline (60 hand-written reviews, 20 per class,
tiled 4×) so the notebook is fully self-contained. It pairs naturally with the sibling AG-News
notebook (§8.17) — same BoW + MLP recipe, different task — and previews the future
`text_classification-imdb-distilbert-hf` transformer counterpart.

## 8.18.1 Problem & motivation

Three-class sentiment classification (negative / neutral / positive) is the canonical
text-classification teaching task. Sentiment has two famously-good recipes, and this notebook runs
both on the same corpus:

1. **VADER** (Hutto & Gilbert, 2014) — a hand-tuned lexicon plus grammar rules. No training. Ships
   with NLTK. A strong, domain-robust baseline; in many production settings it beats supervised
   neural models on out-of-distribution text.
2. **Supervised neural** — spaCy lemmatize → 100-token BoW → 32-hidden MLP, trained on labeled
   examples. Wins on in-distribution text; loses on domain shift.

The notebook exists for three reasons:

1. **Make the "lexicons are competitive" lesson concrete.** Sentiment is the task where the gap
   between a zero-training lexicon and a learned neural model is smallest and most
   production-relevant. The recorded head-to-head on the embedded corpus (VADER 81.25% vs neural
   MLP 79.17%) makes the point on real numbers, not hand-waving.
2. **Land the VADER contract end-to-end.** The `SentimentIntensityAnalyzer` API, the compound-score
   threshold mapping (`±0.05` per the original paper), the lazy `nltk.download('vader_lexicon')`
   fallback — these are all first-class and reproducible. A reader who copies this half of the
   notebook has a production-ready sentiment baseline in ten lines.
3. **Reuse the §8.17 recipe on a different task.** The spaCy BoW + `FeedFwdNN` MLP half is
   identical in shape to the AG-News notebook (same train-only vocab, same L2-normalized BoW,
   same `NNModel` factory). The contrast — same learned recipe, different baseline — isolates
   what the baseline choice buys.

The falsifiable hypothesis tested by the notebook is that on a 240-review embedded corpus the
neural MLP *does not* meaningfully outperform VADER — i.e. that the "always include the lexicon
baseline" advice is load-bearing at this scale, not a courtesy.

## 8.18.2 Concepts

| Concept | Where it shows up |
|---|---|
| 3-class sentiment classification | negative / neutral / positive; one correct label per review |
| VADER lexicon + polarity rules | `SentimentIntensityAnalyzer().polarity_scores(text)['compound']` |
| Compound-score threshold mapping | `> +0.05` → positive, `< -0.05` → negative, else neutral (paper defaults) |
| Rule-based vs learned classifier | The notebook's central contrast — zero-training lexicon vs trained MLP |
| spaCy BoW featurizer | Same recipe as §8.17: lemmatize + train-only vocab + L2-normalized counts |
| MLP classifier | `nnx.FeedFwdNN`, `hidden_dims=[32]`, ReLU, dropout 0 |
| Softmax + cross-entropy | The neural training objective over the three logits |
| Stratified train/test split | `train_test_split(..., stratify=labels_np)` keeps classes balanced |
| Lazy NLTK resource download | `try / except LookupError → nltk.download('vader_lexicon')` |

The `nnx` surface consumed is the same tabular/text-classification surface as §8.17 (`NNModel`,
`NNParams`, `NNModelParams`, `NNTrainParams`, `NNOptimParams`, the enums). The VADER half sits
entirely in NLTK (`nltk.sentiment.vader.SentimentIntensityAnalyzer`); the glue is a thin
`vader_predict(text)` function. `prettytable.PrettyTable` renders the head-to-head comparison.

## 8.18.3 Mathematical formulation

The two recipes produce the same output (a label in {neg, neu, pos}) by very different math.

**VADER (rule-based).** VADER maintains a hand-curated lexicon mapping word/emoji stems to a
valence in \([-4, +4]\). For an input text it sums the valences, applies grammar-rule
adjustments (intensifiers like *very*, negators like *not*, punctuation boost like *!!!*), and
normalizes to a `compound` score in \([-1, +1]\):

\[
\mathrm{compound}(x) = \mathrm{normalize}\!\left(\sum_{w \in x} \mathrm{valence}(w) \cdot \mathrm{rules}(x, w)\right) \in [-1, +1].
\]

The `normalize` step is a fixed algebraic squash (sum divided by the square root of sum-of-squares
plus an alpha constant, then a tanh-like transform) — details in the VADER paper. The notebook
then thresholds:

\[
\hat{y}(x) =
\begin{cases}
\text{positive} & \text{if } \mathrm{compound}(x) > +0.05, \\
\text{negative} & \text{if } \mathrm{compound}(x) < -0.05, \\
\text{neutral} & \text{otherwise.}
\end{cases}
\]

The thresholds `±0.05` are the original-paper defaults; strict inequalities mean exactly `±0.05`
falls into neutral. Domain-specific tuning (e.g. `> 0.2` for more conservative positive
classification) helps on noisy text; the notebook keeps the defaults for reproducibility.

**Neural MLP (learned).** Identical math to §8.17.3 — L2-normalized BoW vector \(x\), single
hidden layer with ReLU, three-class softmax + cross-entropy:

\[
z = W_2 \, \mathrm{ReLU}(W_1 x + b_1) + b_2, \qquad
\mathcal{L}(z, c) = -\log \frac{e^{z_c}}{\sum_j e^{z_j}},
\]

with \(W_1 \in \mathbb{R}^{32 \times 100}\), \(W_2 \in \mathbb{R}^{3 \times 32}\). Adam with
\(\eta = 5 \times 10^{-3}\), weight decay \(10^{-3}\), momentum \((0.9, 0.999)\).

The key contrast is *not* the math but the *training cost*: VADER pays zero (the lexicon ships
with NLTK); the neural MLP pays 192 samples × 60 epochs of gradient descent. On a 240-review
corpus the lexicon's prior knowledge is worth roughly the learned model's capacity to overfit the
train distribution — which is exactly the lesson.

## 8.18.4 Architecture

![Feed-forward MLP](../diagrams/img/mlp.png)

Two independent pipelines, evaluated on the same test split.

**VADER pipeline (frozen):** `SentimentIntensityAnalyzer` → `compound` score → `±0.05` threshold
mapping → label. No learned parameters, no training step, deterministic given the lexicon.

**Neural pipeline (`Nets.FEED_FWD`):** identical shape to the AG-News MLP, one notch smaller:

| Layer | Width | Activation |
|---|---|---|
| Input | 100 (vocab size) | — |
| Hidden | 32 | ReLU |
| Output | 3 (sentiment classes) | softmax (via cross-entropy) |

The shared neural contract:

- **Net:** `Nets.FEED_FWD`
- **Loss:** `Losses.CROSS_ENTROPY`
- **Optimizer:** `Optims.ADAM`, `max_lr=5e-3`, `momentum=(0.9, 0.999)`, `weight_decay=1e-3`
- **Device:** `Devices.CPU`
- **Epochs:** `60` (full run) or `5` (`SMOKE_TEST=1`)
- **Batch size:** `16`
- **Seed:** `0`

The data: 60 unique reviews (20 per class) tiled `CORPUS_REPEAT=4` → 240 documents, split 80/20
stratified → 192 train / 48 test (16 per class in test). Each class is a mix of movie, product,
restaurant, and miscellaneous factual reviews — the factual reviews (release dates, weights,
opening hours) are the neutral class and the weak spot for both recipes.

## 8.18.5 Code walkthrough

### VADER setup with lazy lexicon download

```python
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _sia_probe = SentimentIntensityAnalyzer()        # forces lexicon lookup
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

The throwaway `_sia_probe = SentimentIntensityAnalyzer()` construction forces the lexicon lookup
*inside* the `try` block, so a missing lexicon raises `LookupError` and triggers the download
rather than failing later at predict time. CI pre-downloads the lexicon (~125 KB) in the
`tier-a-papermill` job; this block is the fresh-local-install fallback.

### VADER prediction

```python
sia = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = sia.polarity_scores(text)['compound']
    if score > COMPOUND_POS_THRESHOLD:   return 2    # positive
    if score < COMPOUND_NEG_THRESHOLD:   return 0    # negative
    return 1                                          # neutral
```

One call per document, no batching, no training. The `compound` score is the normalized summary
valence; the two thresholds map it to the three labels. This is the entire VADER pipeline.

### Neural half (spaCy BoW + MLP)

```python
def tokenize(text):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc
            if not t.is_stop and not t.is_punct and t.lemma_.strip()]

counter = Counter()
for i in train_idx:                                  # train-only vocab (cf. §8.17)
    counter.update(tokens_per_doc[i])
vocab = {tok: idx for idx, (tok, _) in enumerate(counter.most_common(VOCAB_SIZE))}

def featurize(toks):
    v = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for tok in toks:
        if tok in vocab: v[vocab[tok]] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

clf = NNModel(net_params=NNParams(input_dim=VOCAB_SIZE, output_dim=len(LABEL_NAMES),
                                  hidden_dims=HIDDEN_DIMS, dropout_prob=0.0,
                                  activation=Activations.RELU),
              params=NNModelParams(net=Nets.FEED_FWD, device=DEVICE, loss=Losses.CROSS_ENTROPY))
run = clf.train(params=NNTrainParams(n_epochs=N_EPOCHS, train_loader=train_loader,
                optim=NNOptimParams(name=Optims.ADAM, max_lr=LR,
                                    momentum=(0.9, 0.999), weight_decay=1e-3)))
```

This is line-for-line the §8.17 recipe with `VOCAB_SIZE=100`, `HIDDEN_DIMS=[32]`, three output
classes. The train-only vocabulary and L2-normalized BoW carry over verbatim — the pitfalls
discussed in §8.17.7 (train-only vocab, `en_core_web_sm` install) apply here unchanged.

### Head-to-head comparison

```python
vader_acc  = accuracy_score(y_test, [vader_predict(t) for t in X_test_text])
neural_acc = accuracy_score(y_test, np.argmax(clf.predict_proba(X_test), axis=1))
# PrettyTable renders the side-by-side accuracy + training-cost row.
```

Both recipes are evaluated on the *same* 48-doc stratified test split; the PrettyTable surfaces
not just the accuracy but the *training cost* — the load-bearing asymmetry the lesson turns on.

## 8.18.6 Results & analysis

On the recorded Tier-A run (`SMOKE_TEST=0`, 60 epochs, 720 iterations) the head-to-head on the
48-doc stratified test split, recorded verbatim:

| Recipe | Accuracy | Training cost |
|---|---|---|
| **VADER (rule-based)** | **81.25%** | none (lexicon ships with nltk) |
| neural MLP [32] | 79.17% | 192 samples × 60 epochs |

VADER wins by ~2 percentage points — on zero training. The neural MLP's final train loss was
`0.1313`, i.e. it fit the train split well; the gap to test is the generalization gap, not an
optimization failure. Per-class metrics, recorded verbatim:

| Recipe | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| VADER | negative | 1.00 | 0.69 | 0.81 |
| VADER | neutral | 0.76 | 0.81 | 0.79 |
| VADER | positive | 0.75 | 0.94 | 0.83 |
| neural | negative | 0.80 | 0.75 | 0.77 |
| neural | neutral | 0.73 | 1.00 | 0.84 |
| neural | positive | 0.91 | 0.62 | 0.74 |

Three observations:

1. **The two recipes fail in complementary ways.** VADER has high positive recall (0.94 — it
   catches every clearly-positive review) but low negative recall (0.69 — it misses reviews whose
   negativity is subtle or context-dependent). The neural MLP is the mirror: perfect neutral
   recall (1.00 — it over-predicts neutral) but low positive recall (0.62). Neither dominates.
2. **Neutral is the weak spot for both, in different senses.** VADER's neutral precision is 0.76
   (it falsely routes negative reviews to neutral); the neural MLP's neutral precision is 0.73
   with recall 1.00 (it absorbs everything into neutral). This is the canonical sentiment failure
   mode: factual "neutral" text (release dates, weights, hours) has no polarity words for VADER
   and weak learned signal for the MLP — both guess neutral by default.
3. **The lesson is the relative ordering, not the absolute numbers.** At 48 test docs each
   misclassification costs ~2 points and the seed moves the boundary; do not read 81.25% vs
   79.17% as a precise gap. Read it as "the lexicon is *competitive* — within striking distance
   of the learned model on zero training, and ahead of it on this split." That is the
   production-relevant claim: a zero-cost baseline that ties the learned model is a baseline you
   must ship before claiming the neural model earns its training cost.

## 8.18.7 Pitfalls & edge cases

- **`vader_lexicon` is a separate download.** `pip install nltk` does not pull the lexicon — it
  ships separately and is fetched via `nltk.download('vader_lexicon')`. CI pre-downloads it in
  the `tier-a-papermill` job; the notebook's `try / except LookupError` block is the fresh-local
  fallback. Without it, the first `SentimentIntensityAnalyzer()` call raises `LookupError`.
- **The `±0.05` thresholds are paper defaults, not optimal.** Domain-specific tuning helps on
  noisy real-world text. Social-media sentiment typically wants `> 0.2 / < -0.2` to avoid
  over-predicting polarity; product reviews sometimes want the opposite. The notebook keeps the
  defaults for reproducibility and flags this in the README.
- **Neutral detection is the weak spot for both recipes.** Embedded "neutral" reviews are factual
  statements (release dates, store hours, package weights). VADER calls these neutral by default
  (no polarity words) — good — but the neural MLP, with only ~16 neutral train samples, sometimes
  overfits to specific neutral keywords and over-predicts neutral (recall 1.00, precision 0.73).
  A bigger neutral-train-set or class-balanced loss would help; neither is in this notebook.
- **Train-only vocabulary applies here too.** The neural half inherits the §8.17.7 footgun
  verbatim: build the vocab from train_idx *only*, or silently inflate test accuracy.
- **The corpus is tiny and embedded.** 60 unique reviews × 4 tiling = 240 documents. Real
  sentiment benchmarks (IMDB 50k, Amazon Reviews) are 200–1000× larger. Absolute accuracy is not
  directly comparable; the *relative ordering* (VADER ≈ neural at this scale) is the pedagogical
  point and the load-bearing lesson.
- **BoW destroys negation scope.** "Not good" and "good" share the lemma *good*; BoW cannot tell
  them apart. VADER handles negation via its grammar rules (`not` flips the valence). This is one
  structural reason VADER stays competitive — its rule layer captures minimal syntax the BoW
  featurizer flattens away.

## 8.18.8 Extensions & references

- **Add a transformer baseline.** Fine-tune a DistilBERT (or the future
  `text_classification-imdb-distilbert-hf` task) on the same split. The point of the comparison
  is to show where the learned model *finally* pulls ahead of the lexicon — and at what training
  cost. On IMDB-50k, DistilBERT reaches ~93% vs VADER's ~75%; on a 240-review embedded corpus the
  gap may not open.
- **Sweep the VADER thresholds.** Grid-search `compound ∈ {0.0, 0.05, 0.1, 0.2, 0.3}` on a held-out
  val split and report the accuracy surface. This is the cheapest win available — the defaults are
  not optimal for most domains.
- **Ensemble VADER + neural.** Average the two probability vectors (VADER's via a softmax over
  the compound-score bins, the neural's directly) and report the blended accuracy. Lexicon +
  learned ensembles are a standard production pattern and often beat either alone by 1–3 points.
- **Scale to a real benchmark.** IMDB 50k or Amazon Reviews via HuggingFace `datasets`. At scale
  the neural model finally overtakes VADER — but the lexicon stays within ~15 points at zero
  training cost, which is the production-relevant headline.
- **Reference reading.** Hutto & Gilbert, "VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis" (2014) for the compound-score normalization and the `±0.05` thresholds; Bird, Klein &
  Loper, *Natural Language Processing with Python* (NLTK book) for the `SentimentIntensityAnalyzer`
  API. The §8.17 references cover the BoW + MLP half.
