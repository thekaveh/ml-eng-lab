# 3. Concepts

This page is the cross-cutting foundations reference for the notebook collection: the training,
evaluation, and generalization machinery that *every* task composes, regardless of paradigm. Each
per-task deep-dive (under `notebooks/`) assumes this vocabulary and reports against these metrics;
this page collects them in one place so the deep-dives can focus on what is *task-specific* (the
architecture, the data pipeline, the loss head) rather than re-deriving softmax or ARI from
scratch each time. The three smallest, cleanest reference points are the Iris MLP
([notebooks/tabular_classification-iris-mlp-pytorch.md](notebooks/tabular_classification-iris-mlp-pytorch.md)),
the MNIST FFNN sweep
([notebooks/image_classification-mnist-ffnn-pytorch.md](notebooks/image_classification-mnist-ffnn-pytorch.md)),
and the TinyShakespeare transformer
([notebooks/text_generation-tinyshakespeare-transformer-pytorch.md](notebooks/text_generation-tinyshakespeare-transformer-pytorch.md)).

## 3.1. Training & optimization

### 3.1.1. The training loop

Training a neural network is the iterative application of one rule: measure the disagreement
between prediction and target with a *loss*, compute the gradient of the loss with respect to each
parameter, and step the parameters a small amount against the gradient. One pass over the full
training set is an **epoch**; one parameter update is a **step**; the set of samples used to
compute a single gradient is a **batch** (or **mini-batch** when it is a subset of the training
set). The Iris MLP trains 300 epochs in mini-batches of 32; the MNIST FFNN notebook runs 500
epochs of *full-batch* SGD (one step per epoch, the same regime as its NumPy sibling); the
TinyShakespeare transformer runs 5 epochs of 43 mini-batches each. Both regimes are first-class.

The `nnx` library factors the loop into configuration objects and a single `.train(...)` call:
`NNModelParams` selects the net family, device, and loss; `NNParams` configures the topology
(`hidden_dims`, `dropout_prob`, `input_dim`, `output_dim`); `NNTrainParams` carries the budget,
the loaders, and the optimizer block (`NNOptimParams`). The returned `NNRun` is the training
history — a list of iteration data points (`run.idps`), each carrying `train_edp` and `val_edp`
(evaluation data points with `error`, `accuracy`, `precision`, `recall`, `f1`, `loss`, `lr`). The
same `NNRun` is the serialization surface: `NNRun.load("best")` restores the best checkpoint from
disk in a fresh session.

### 3.1.2. Loss functions

The **loss** is the scalar the optimizer minimizes. Three losses cover every supervised task; a
fourth (smooth-L1 reconstruction) covers the self-supervised case.

**Cross-entropy (CE)** is the multiclass classification loss. The classifier produces logits
\(z = (z_0, \dots, z_{K-1})\); softmax maps them to a probability simplex, and CE measures the
disagreement with the one-hot label \(y\):

\[
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
\mathcal{L}(z, y) = -\sum_i y_i \log \hat{y}_i = -\log \hat{y}_{c},
\]

where \(c\) is the correct-class index. The Iris and MNIST notebooks select this via
`Losses.CROSS_ENTROPY`; the framework pairs softmax with CE internally so neither notebook
instantiates either by hand. For autoregressive language modeling, the same CE applies along the
vocab dimension of every position. The TinyShakespeare transformer emits logits of shape
\((B, T, V)\); the custom `lm_train_step` flattens the batch and sequence axes into one "token"
axis (\((B \cdot T, V)\)) so `torch.nn.functional.cross_entropy` applies along the vocab dimension:

\[
\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log P(x_{t+1}=x_{t+1} \mid x_{\le t}).
\]

The `train_step_fn=` hook on `model.train(...)` is the seam that swaps in this LM-specific loss;
the same seam injects MSE for autoencoders, smooth-L1 for I-JEPA, and the contrastive DPO loss for
preference alignment.

**Binary cross-entropy (BCE)** drives the binary case — sentiment classification and link
prediction. The Karate link-prediction notebook scores an edge as the dot product of endpoint
embeddings and applies `binary_cross_entropy_with_logits` on the raw scores against labels \(1\)
(positive edge) and \(0\) (sampled negative):

\[
\mathcal{L}_{\text{BCE}} = -\frac{1}{|E^{+}|+|E^{-}|}\!\!\!\sum_{(u,v)\in E^{+}\cup E^{-}}\!\!\!\Big[ y_{uv}\log\sigma(z_u^{\top} z_v) + (1-y_{uv})\log\!\big(1-\sigma(z_u^{\top} z_v)\big) \Big].
\]

Using `with_logits` (rather than sigmoid then `BCELoss`) keeps the computation in the numerically
stable log-sum-exp regime.

**Mean-squared error (MSE)** is the regression loss. The Diabetes MLP produces a scalar
\(\hat{y} \in \mathbb{R}\) and minimizes

\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \bigl(y_i - \hat{y}_i\bigr)^2,
\]

via `Losses.MEAN_SQUARED_ERROR` paired with a one-unit output head. The same reconstruction
objective — `F.mse_loss(net(X), X)` — trains the autoencoders used for dimensionality reduction and
clustering, where the input *is* the target. MSE expects float targets shaped \((N, 1)\) to match
the network output; passing \((N,)\) broadcasts and silently produces a different reduction.

### 3.1.3. Gradient descent → SGD → Adam

**Vanilla gradient descent** updates each parameter \(\theta\) by stepping against the gradient,
\(\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)\), where \(\eta\) is the
learning rate. **Mini-batch SGD** approximates the gradient over a batch rather than the full
dataset, which is what makes the per-step cost tractable. The from-scratch NumPy MNIST notebook
implements this update directly — every weight and every gradient is visible in NumPy; the
PyTorch-via-`nnx` sibling swaps the same loop onto Adam.

**Adam** (`Optims.ADAM`) is the optimizer the `nnx` notebooks actually use. It maintains per-
parameter first and second moment estimates of the gradient and steps using their bias-corrected
ratio:

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad
\theta_t \leftarrow \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\]

Typical settings: \(\eta \in \{10^{-2}, 3 \times 10^{-4}, 5 \times 10^{-3}, 5 \times 10^{-4}\}\)
(task-tuned); \((\beta_1, \beta_2) = (0.9, 0.999)\) for feed-forward tasks and \((0.9, 0.95)\) for
the transformer (the smaller \(\beta_2\) is the GPT-family convention, which decays the
second-moment estimate faster and is more forgiving on early-step loss spikes); weight decay
\(5 \times 10^{-4}\) (Iris, Karate) or \(5 \times 10^{-5}\) (MNIST) or \(0\) (transformer). The
transformer also sets `grad_clip_norm=1.0` to clip the global gradient norm — a stabilization that
tames the early-step spikes.

### 3.1.4. Learning-rate schedules

The MNIST FFNN notebook uses `ReduceLROnPlateau` as its default scheduler — it watches validation
error and multiplies the learning rate by `factor = 0.95` after `patience = 8` iterations of
insufficient improvement (delta below `threshold = 0.001`), enforces a `cooldown = 2`-iteration
pause between reductions, and floors at `min_lr = 1e-7`. The effect is *per-run self-annealing*:
each sweep candidate anneals independently as its own validation curve plateaus, so no global LR
schedule has to be tuned by hand. The LR-overlay plot the MNIST notebook produces
(`VisUtils.multi_line_plot` on `idp.lr`) makes this visible — the step-downs cluster across the top
runs at the same iteration range, the visible signature of the scheduler doing its job. The other
notebooks mostly pin a flat LR — appropriate for the short Tier-A budgets where the scheduler's
overhead would not pay back.

### 3.1.5. The `nnx` training-loop shape

The shape the notebooks converge on, repeatedly:

| Object | Role |
|---|---|
| `NNModelParams` | Net family, device, loss — the "what" |
| `NNParams` | Topology (`hidden_dims`, `dropout_prob`, dims) — the "how big" |
| `NNOptimParams` | Optimizer (`name`, `max_lr`, `momentum`, `weight_decay`, `grad_clip_norm`) — the "how fast" |
| `NNTrainParams` | `n_epochs`, loaders, seed — the "how long" |
| `NNModel.train(params=...)` | The single call that runs the loop |
| `NNRun` | Returned training history (`idps`, checkpoints) |
| `train_step_fn=` (optional) | Seam for swapping the default classification step for LM / AE / JEPA / DPO |

The default step assumes a classification head — forward through the net, compute CE against the
batch labels, backprop, step the optimizer, return an `NNEvaluationDataPoint`. The `train_step_fn=`
hook is the load-bearing customization seam: the transformer passes an `lm_train_step` that
flattens \((B, T, V)\) for cross-entropy; the autoencoder and clustering notebooks pass a step
computing `F.mse_loss(net(X), X)`; the I-JEPA notebook passes the factory-built
`jepa_train_step_factory(...)`. The framework still owns the scheduler, the checkpoint cadence, and
the per-iteration bookkeeping — only the loss computation is swapped.

## 3.2. Evaluation & metrics

Training loss going down proves the optimizer is working; *held-out* metrics prove the model is
learning something that generalizes. The collection's evaluation surface splits along two axes:
*what kind of curve* to plot, and *which metric* summarizes the held-out score for a given task
family.

### 3.2.1. Curves and confusion matrices

The canonical training diagnostic is the **per-iteration loss (or error) curve**, overlaid for
training and validation. `NNRun.idps` is the data source; `VisUtils.multi_line_plot` is the
standard renderer (taking `x`, `yss`, and `yss_legend`). The MNIST sweep overlays the top five
candidates on one figure so the capacity-vs-marginal-gain curve is directly readable; the Iris
notebook overlays all three candidates so the linear-vs-MLP gap is visible at a glance.

For classification, the **confusion matrix** is the directly-interpretable artifact: each row is
the true class, each column the predicted class, and the diagonal-to-off-diagonal ratio answers
"which class pairs does this model confuse?" The Iris confusion matrices make the
*versicolor*/*virginica* overlap visible as the only meaningfully non-zero off-diagonal;
`VisUtils.confusion_matrix` is the renderer (`normalize=False` for raw counts, `normalize=True`
for row-normalized recalls).

### 3.2.2. Per-family metrics

Each task family has its own metric. The collection deliberately does *not* flatten everything to
accuracy — the right metric for a regression is \(R^2\), for a clustering is ARI, for a link
prediction is AUC, and reporting accuracy for any of them would be meaningless.

| Task family | Metric | Where it lands |
|---|---|---|
| Multiclass classification | Accuracy, macro precision/recall/f1 | Iris MLP, MNIST FFNN, MoE, PEFT |
| Regression | \(R^2\), RMSE, MAE, MSE | Diabetes MLP |
| Language modeling | Per-token cross-entropy, perplexity, bits-per-character | TinyShakespeare transformer |
| Link prediction | ROC-AUC, Average Precision | Karate GraphSAGE |
| Clustering / community detection | Adjusted Rand Index (ARI), Normalized Mutual Info (NMI) | Iris KMeans-vs-AE, Karate Louvain-vs-GNN |
| Self-supervised representation | Linear-probe accuracy | I-JEPA on Fashion-MNIST |

**Multiclass classification.** Accuracy is the fraction of correctly predicted labels. Macro-
averaged precision, recall, and f1 compute each metric *per class* and then average equally — the
interpretable choice under any class balance, and numerically identical to micro-averaging under
uniform balance. The Iris notebook reports all four; the candidate ranking keys off f1-macro
(then accuracy) because per-class behavior at the *versicolor*/*virginica* boundary is the
variation that actually distinguishes candidates.

**Regression.** The coefficient of determination \(R^2\) is the fraction of target variance
explained, relative to the constant-mean predictor:

\[
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}.
\]

\(R^2 = 1\) is a perfect predictor; \(0\) is "no better than predicting the mean"; \(< 0\)
(possible on a held-out test split) flags "worse than the mean." RMSE \(= \sqrt{\mathcal{L}}\)
returns the error to target units; MAE \(= \frac{1}{N}\sum_i |y_i - \hat{y}_i|\) is the
robust-to-outliers sibling. The Diabetes notebook reports all four because each brackets a
different failure mode — a negative \(R^2\) flags a broken model, an \(\mathrm{MAE} \ll
\mathrm{RMSE}\) flags heavy-tailed residuals.

**Language modeling.** The directly-recorded LM loss is the per-batch cross-entropy (sum or mean
over the \(B \times T\) token positions, depending on the step's reduction). To compare against
published numbers, divide by \(B \times T\) for a *per-token* mean, then exponentiate for
**perplexity** (\(\mathrm{ppl} = e^{\mathcal{L}_{\text{mean}}}\)) or divide by \(\ln 2\) for
**bits-per-character** (\(\mathrm{bpc} = \mathcal{L}_{\text{mean}} / \ln 2\)). The TinyShakespeare
notebook's 63 → 7.5 trajectory is the *per-batch* number — load-bearing as evidence the loop is
correct, but not directly comparable to a published perplexity without the per-token normalization.

**Link prediction.** ROC-AUC and Average Precision score the ranking of held-out edges by their
predicted score. AUC is the probability a random positive ranks above a random negative; AP is
the precision-recall area. The Karate notebook's 0.735 val AUC (well above 0.5) is the
load-bearing "the encoder learned real link signal" evidence; the 0.431 test AUC on 30 test edges
is honest variance, not a defect — and AP (0.579) is the more stable metric on a small,
class-imbalanced ranking.

**Clustering / community detection.** ARI and NMI score the agreement between the predicted
clustering and true labels *post hoc* — the labels never enter training. ARI corrects the Rand
Index for chance:

\[
\mathrm{ARI} = \frac{\binom{N}{2} \cdot \mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]},
\]

so \(1\) is perfect, \(0\) is random, negative is worse than chance. NMI normalizes mutual
information to \([0, 1]\) regardless of cluster count: \(\mathrm{NMI}(U, V) = 2 I(U; V) / (H(U) +
H(V))\). Both are symmetric in their arguments and invariant under permutations of cluster labels —
essential properties for a clustering metric. The Iris KMeans-vs-AE notebook reports both (raw
features ARI 0.716, AE latent ARI 0.835); the win on the learned latent confirms the reconstruction
objective has implicitly grouped similar inputs.

**Self-supervised representation.** The standard SSL evaluation is **linear-probe accuracy**:
freeze the pretrained encoder, precompute its embeddings once, train a single `Linear(d_model → C)`
layer on those frozen features, report probe accuracy on the held-out label set. The I-JEPA
notebook's 74.43% probe accuracy (against a 10% random baseline and a ~93% supervised ceiling) is
the proof of life that *unlabeled* pretraining produced non-trivial features. Mean-pooling over
patch embeddings is the simplest probe input and a *lower bound* on representation quality — a more
expressive probe (attention pooling, fine-tuned head) would report higher accuracy from the same
frozen features.

### 3.2.3. Model selection

For sweep-shaped notebooks (MNIST's 18 candidates, Iris's 3, Diabetes's 2 MLPs), the selection
metric is **best-iteration validation error** — the lowest validation error a run ever achieved,
*not* its final-epoch value:

\[
\mathrm{valerr}_{\text{best}}(r) = \min_{t \in r.\mathrm{idps}} \mathrm{val\_edp}.\mathrm{error}.
\]

Ranking by best-iteration is robust to late-training overfitting oscillations; the selected
checkpoint corresponds to the best-iteration moment. This is appropriate but worth flagging when
comparing against papers that report final-epoch numbers — best-iteration is an *optimistic*
estimate of generalization, and a stricter regime (k-fold cross-validation, explicit early
stopping) tightens the variance estimate at the cost of compute.

## 3.3. Generalization & regularization

The training loss going to zero is *not* the goal — a model that fits the training set perfectly
often generalizes poorly. The collection's generalization regime is a small set of disciplines:
hold out data the model never sees, regularize the parameters, and seed everything so the result is
reproducible.

### 3.3.1. Overfitting and the train/val/test split

**Overfitting** is the gap between train and validation performance — the model fitting noise in
the training set as if it were signal. The Diabetes notebook is the cleanest example: the deeper
`[32, 16]` MLP (625 params) *underperforms* the smaller `[8]` MLP (89 params) on the held-out test
split because it overfits the 308-sample train split before it extracts any non-linear signal that
generalizes. Capacity is not monotonically helpful — at small-data regimes, more parameters can
mean *worse* generalization.

The defense is the **train/val/test split**: the model trains on train, model selection uses val,
and the reported headline is measured once on test, which the model never saw during training or
selection. The collection's standard split is 70/15/15 (Iris, Diabetes); the MNIST notebooks use
the canonical 60k/1k/9k partition that `NNDataset` exposes by default. **Stratification** matters
for classification at small sample counts — at 150 Iris samples times 15% test, a default random
split can plausibly miss a whole class, so `stratify=` on `train_test_split` is load-bearing. It
is *inapplicable* for regression (continuous targets) and *unnecessary* for unsupervised
clustering (no split at all — all 150 Iris samples are seen by both KMeans and the AE).

### 3.3.2. Leakage footguns

The collection documents four leakage patterns that silently inflate metrics if missed:

- **Scaler fit on train only.** Applying `fit_transform` to val or test leaks their feature ranges
  into training. The canonical anti-leakage pattern is `scaler.fit_transform(X_train)` then
  `scaler.transform(X_val)` and `scaler.transform(X_test)`. Both Iris (MinMaxScaler) and Diabetes
  (StandardScaler) notebooks follow this; both flag it in their pitfalls sections.
- **Link-split direction leakage.** On an undirected graph, both directions of the same edge can
  land in different splits unless `RandomLinkSplit(is_undirected=True)` is set. Without it, the
  model has effectively seen each test edge's reverse during training, and test AUC silently
  inflates. The Karate link-prediction notebook pins this explicitly.
- **Message-passing vs supervised-edge confusion.** In link prediction, `edge_index` (the
  message-passing edges the encoder aggregates over) is *always* available; `edge_label_index`
  (the supervised BCE positives) is train-only. Aggregating over `edge_label_index` at train time,
  or evaluating over `edge_index`, are both subtle leaks. The Karate notebook's pitfalls section
  flags this as the load-bearing distinction.
- **Autoregressive next-token leakage.** The LM target is the input rolled by one
  (`y = x.roll(-1)`) — getting this wrong (e.g., `y = x` or `y = x.roll(1)`) silently lets the
  model see the token it is supposed to predict. The TinyShakespeare notebook uses the canonical
  left-shift; the same trick recurs in every autoregressive LM.

### 3.3.3. Dropout, weight decay, and early stopping

**Dropout** zeros each hidden unit with probability \(p\) (`dropout_prob`) per forward pass and
scales the survivors by \(1/(1-p)\) to keep the expected activation constant; at test time it is
disabled. The effect is implicit ensembling — each forward pass sees a thinned network. The Iris
MLP uses `dropout_prob=0.1` (already at the upper end for a 50-parameter model on 105 samples;
heavier dropout starves the hidden layer); the MNIST sweep covers `dropout_prob ∈ {0.25, 0.5}`;
the transformer, autoencoder, and link-prediction encoder set `dropout_prob=0.0`. The right value
is task-tuned, not a constant.

**Weight decay** (L2 regularization on the parameters, folded into the optimizer step) pulls
weights toward zero proportional to a coefficient \(\lambda\), penalizing the large weights that
are the signature of a model memorizing individual training examples. The collection uses
\(\lambda = 5 \times 10^{-4}\) (Iris, Karate), \(5 \times 10^{-5}\) (MNIST), \(10^{-4}\) (Diabetes,
I-JEPA pretrain), and \(0\) (transformer, autoencoder) — small models on small data benefit from
more decay; the transformer's short-budget, low-LR regime does not need it.

**Early stopping** monitors a validation metric and halts training when it stops improving,
recovering the overfit-to-train gap without manual epoch tuning. `nnx` exposes `EarlyStopping`
(the Diabetes notebook flags that its default `monitor="val_edp.error"` is the *classification*
monitor; regression must pass `monitor="val_edp.loss"` explicitly), but most Tier-A notebooks use
a fixed `n_epochs` for budget predictability in CI and rely on **best-iteration validation
ranking** (§3.2.3) as the implicit early-stopping signal — the selected checkpoint is the one at
the best-iteration moment, not the final epoch.

### 3.3.4. Seeding and reproducibility

Reproducibility is the discipline that makes a recorded number *mean something* — without it, the
headline metric is a single draw from a wide distribution. The collection's convention is
`nnx.set_seed(0)`, which pins Python's `random`, NumPy's RNG, and PyTorch's CPU + CUDA + cuDNN
RNGs in one call. For controlled-comparison sweeps (Iris, MNIST, Diabetes), the seed is
**re-pinned inside the candidate loop** so identical seeds produce identical weight inits — the
*only* difference between candidates is the swept axis (`hidden_dims`, `dropout_prob`). Without
the re-pin, the RNG state advances between candidates and the inits differ for reasons unrelated
to the swept axis, which defeats the controlled-comparison framing.

External RNGs — `sklearn.train_test_split(random_state=...)`, `KMeans(random_state=...)`,
`RandomLinkSplit`'s seed, `negative_sampling`'s draws — are pinned separately. The link-prediction
notebook is honest about a residual variance source it does *not* pin: it uses a single seed, so
the recorded 0.431 test AUC is one draw from a wide band, and the pitfalls section flags averaging
across seeds as the robust estimate. The same honesty recurs in the clustering notebook (single
seed, single split) and the Iris notebook (single 70/15/15 split, "on this split" rather than "on
Iris"). Reproducibility is the floor; *honesty about the residual variance* is the discipline.

Device determinism is the related caveat. `Devices.get()` (used by the MNIST notebook) returns
CUDA if available else CPU, which means the same notebook produces different wall-clock times (and
slightly different floating-point results, due to non-deterministic GPU kernels) across machines.
Reproducibility *across machines* requires pinning `Devices.CPU` explicitly — which the MNIST
notebook does not do, trading cross-machine bit-exactness for the speedup when a GPU is present.
