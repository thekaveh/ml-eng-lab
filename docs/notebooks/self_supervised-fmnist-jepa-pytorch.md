# 8.11 Self-supervised — I-JEPA on Fashion-MNIST

A comprehensive walk-through of `notebooks/self_supervised-fmnist-jepa-pytorch/` — the
in-repo exercise of the `nnx` I-JEPA stack (`ViTNN`, `build_target_encoder`, `JEPAPredictor`,
`random_block_mask`, `jepa_train_step_factory`). This page is the deep-dive companion to the
task notebook: it states the problem, builds the math, dissects the architecture, reads the
code top to bottom, reports the measured results, and catalogues the pitfalls and extensions
that govern the recipe.

The notebook is **Tier-A** — CPU re-runs in roughly ninety seconds (the heaviest Tier-A
notebook in the collection) and it is re-executed end-to-end in CI on every pull request. It is
the canonical demo of predictive self-supervised representation learning: a Vision Transformer
context encoder learns by *predicting* the embeddings of a held-out target block produced by a
slow EMA target encoder, with no labels, no view augmentations, and no negative samples. After
pretraining, the encoder is frozen and a single linear layer is trained on Fashion-MNIST
labels to quantify the learned representation's quality — the standard SSL linear-probe
evaluation.

## 8.11.1 Problem & motivation

I-JEPA (Image Joint-Embedding Predictive Architecture, Assran et al., 2023) is Yann LeCun's
framework for self-supervised representation learning. Unlike contrastive methods (SimCLR,
BYOL) which compare augmented views of the same image against negatives, JEPA is **predictive
in embedding space**: given a context block of patches, predict the embeddings of a held-out
target block produced by a frozen EMA copy of the context encoder. No view augmentations, no
negative samples, no collapse-prevention tricks beyond the slow EMA target. The bet is that
prediction in representation space — rather than in pixel space — pushes the encoder to learn
semantically meaningful features without the augmentation-heavy baggage of contrastive SSL.

The `nnx` megamerge ships the full I-JEPA stack as five composable primitives:

- **`ViTNN(image_size, patch_size, in_channels, d_model, n_layers, n_heads)`** — the context
  encoder. Patchifies the image and runs a small ViT.
- **`build_target_encoder(context_encoder)`** — deep-copies the context encoder, freezes it,
  and registers it for EMA updates.
- **`JEPAPredictor(embed_dim, n_patches, predictor_dim, n_layers, n_heads)`** — the small
  transformer that predicts target-patch embeddings from context-patch embeddings.
- **`random_block_mask(n_patches, grid_size)`** — a sampler that splits patches into context
  and target sets per image.
- **`jepa_train_step_factory(target_encoder, predictor, mask_fn, ema_momentum)`** — wires it
  all together, including the EMA update on the target encoder.

This notebook exists for three reasons:

1. **End-to-end exercise of the megamerge JEPA + ViT stack** — the canonical in-repo demo that
   all five primitives work together.
2. **Dataset substitution down to CPU-feasible scale.** The original I-JEPA paper uses ImageNet
   (224² RGB); the `nnx` example uses CIFAR-10 (32² RGB). This notebook uses Fashion-MNIST
   (28² grayscale) because (a) it is already in the collection (no new download), (b) it is
   CPU-feasible at the Tier-A budget, and (c) the plumbing is identical — only `image_size`,
   `in_channels`, and `patch_size` change.
3. **Visible SSL proof-of-life.** The notebook does not stop at "pretrain loss decreases." It
   runs a linear probe on the frozen encoder and reports val accuracy against the 10% random
   baseline — the operational definition of "the encoder learned non-trivial features."

The falsifiable hypothesis tested by the notebook is that, at a deliberately short pretrain
budget (two epochs), the JEPA loss decreases monotonically (the predictor learns to map context
to target embeddings) and the frozen-encoder linear probe lands well above the 10% random
baseline — the proof of life for self-supervised representation learning — while remaining
well below the supervised Fashion-MNIST ceiling.

## 8.11.2 Concepts

| Concept | Where it shows up |
|---|---|
| Self-supervised representation learning | Phase 1 pretrain ignores labels; Phase 2 probe uses them |
| Joint-Embedding Predictive Architecture | Predict target embeddings from context embeddings, both in encoder output space |
| Vision Transformer (ViT) | `ViTNN(image_size=28, patch_size=4, in_channels=1, d_model=64, n_layers=2, n_heads=4)` |
| Patchification | 28×28 → 7×7 = 49 patches at `patch_size=4` |
| Random block masking | `random_block_mask` carves a rectangular target block; the rest is context |
| EMA target encoder | `build_target_encoder` deep-copy + freeze; EMA-updated every step |
| Stop-gradient on target | Targets are detached — gradients flow only through context encoder + predictor |
| Linear probe evaluation | Freeze encoder; train `Linear(d_model → 10)` on mean-pooled patch embeddings |
| `.net` substitution | `NNModel` shell with placeholder `FeedFwdNN`, then `model.net = ViTNN(...)` |
| Reproducibility | `nnx.set_seed(0)` |

The `nnx` surface consumed is `ViTNN`, `JEPAPredictor`, `build_target_encoder`,
`jepa_train_step_factory`, `random_block_mask`, `NNModel`, `NNDataset`, `NNModelParams`,
`NNParams`, `NNTrainParams`, `NNOptimParams`, `Activations`, `Devices`, `Losses`, `Nets`,
`Optims`, and `set_seed`. The predictor is registered as `model.net._jepa_predictor` so the
`NNModel` optimizer picks up its parameters jointly with the context encoder's.

## 8.11.3 Mathematical formulation

The image is patchified into a grid of \(N = 49\) patches (7×7 at `patch_size=4`). Each patch
is linearly embedded into a \(d_{\text{model}}\)-dimensional token; positional embeddings are
added. Let the full patch-token set be
\(\{p_1, \ldots, p_N\}\), \(p_j \in \mathbb{R}^{d_{\text{model}}}\).

`random_block_mask` partitions the patch indices into a *context* set \(\mathcal{C}\) and a
*target* set \(\mathcal{T}\), where \(\mathcal{T}\) is a rectangular block (scale and aspect
sampled per the I-JEPA paper). The context encoder \(f_\phi\) sees only \(\mathcal{C}\); the
target encoder \(f_\xi\) (an EMA copy of \(f_\phi\), frozen to gradients) sees only
\(\mathcal{T}\).

The predictor \(g_\psi\) takes the context-encoder output plus target-position tokens and
predicts the target embeddings:

\[
\hat{z}_j = g_\psi\!\left(f_\phi(x_{\mathcal{C}}),\, \text{pos}_j\right),
\qquad j \in \mathcal{T}.
\]

The training objective is a smooth-L1 (or L2) distance between predicted and *stopped-gradient*
target embeddings:

\[
\mathcal{L}(\phi, \psi)
= \sum_{j \in \mathcal{T}}
\big\lVert \hat{z}_j - \text{sg}\!\left(f_\xi(x_{\mathcal{T}})_j\right) \big\rVert.
\]

The stop-gradient \(\text{sg}(\cdot)\) is essential: without it, the predictor and target
encoder could trivially agree by collapsing both to a constant. Combined with the EMA update,

\[
\xi \leftarrow \tau\, \xi + (1 - \tau)\, \phi,
\qquad \tau = \texttt{ema\_momentum} = 0.996,
\]

the target encoder moves slowly enough that the prediction target is effectively stationary
within a step, preventing trivial collapse. Lower momentum values (0.99 or below) let the
target chase the context and the loss collapses to zero in degenerate ways.

After pretraining, the encoder is frozen and a single linear layer is trained on
mean-pooled patch embeddings for the downstream classification task:

\[
\bar{z} = \frac{1}{N} \sum_{j=1}^{N} f_\phi(x)_j,
\qquad
\hat{y} = \text{softmax}\!\left(W_{\text{probe}}\, \bar{z}\right),
\]

with \(W_{\text{probe}} \in \mathbb{R}^{C \times d_{\text{model}}}\) the only trainable
parameters. Probe accuracy on the frozen features is the standard SSL evaluation.

## 8.11.4 Architecture

The context encoder is a small ViT:

| Stage | Shape | Role |
|---|---|---|
| Input | 1 × 28 × 28 | Grayscale Fashion-MNIST image |
| Patchify | 49 tokens × 64 | 7×7 grid at `patch_size=4`, linear projection to `d_model=64` |
| Transformer | 2 layers, 4 heads | Self-attention over the 49 context patches |
| Output | 49 × 64 | Per-patch embeddings |
| Parameters | — | **102,720** |

The predictor is a tiny transformer:

| Stage | Shape | Role |
|---|---|---|
| Input | context embeddings + target-position tokens | Predict target embeddings from context |
| Transformer | 2 layers, 2 heads, `predictor_dim=32` | Small by design — capacity bottleneck on the predictor |
| Output | \(\lvert\mathcal{T}\rvert\) × 64 | Predicted target embeddings |
| Parameters | — | **30,368** |

The target encoder is a deep-copy of the context encoder (102,720 params, frozen, EMA-updated).
Total trainable parameters across the pretrain step: 102,720 (context encoder) + 30,368
(predictor) = 133,088.

The shared training contract:

- **Context encoder:** `ViTNN(image_size=28, patch_size=4, in_channels=1, d_model=64, n_layers=2, n_heads=4)`.
- **Target encoder:** `build_target_encoder(context)` — deep-copy, frozen, EMA-updated.
- **Predictor:** `JEPAPredictor(embed_dim=64, n_patches=49, predictor_dim=32, n_layers=2, n_heads=2)`.
- **Mask sampler:** `random_block_mask` with `grid_size=7` (defaults: scale `[0.15, 0.20]`, aspect `[0.75, 1.50]`).
- **Phase 1 pretrain optimizer:** `Optims.ADAM`, `max_lr=5e-4`, `momentum=(0.9, 0.999)`, `weight_decay=1e-4`.
- **Phase 2 probe optimizer:** `Optims.ADAM`, `max_lr=1e-2`, `momentum=(0.9, 0.999)`, `weight_decay=0.0`.
- **Device:** `Devices.CPU`.
- **Pretrain epochs:** `2` (full run) or `1` (`SMOKE_TEST=1`).
- **Probe epochs:** `5` (full run) or `1` (`SMOKE_TEST=1`).
- **EMA momentum:** `0.996`.
- **Batch size:** `128`.
- **Seed:** `0`.

The *a priori* expectation: the JEPA prediction loss should fall sharply (the predictor has a
well-defined target), and the frozen-encoder linear probe should land well above the 10%
random baseline but well below the supervised Fashion-MNIST ceiling — the proof-of-life band
for short-budget SSL.

## 8.11.5 Code walkthrough

### Encoder + target + predictor wiring

```python
model.net = ViTNN(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                  in_channels=IN_CHANNELS, d_model=D_MODEL,
                  n_layers=N_LAYERS, n_heads=N_HEADS).to(model.device)
target_encoder = build_target_encoder(model.net)
predictor = JEPAPredictor(embed_dim=model.net.d_model, n_patches=model.net.n_patches,
                          predictor_dim=PREDICTOR_DIM, n_layers=2, n_heads=2).to(model.device)
model.net.add_module("_jepa_predictor", predictor)
```

Registering the predictor *under* `model.net` (rather than as a free-floating attribute) is
load-bearing: the `NNModel` optimizer collects parameters from `model.net`, so attaching the
predictor there means a single `.train()` call updates both the context encoder and the
predictor jointly. The EMA update is name-keyed against the target encoder's parameters, so
the extra predictor parameters are skipped automatically.

### Mask sampler

```python
grid_size = IMAGE_SIZE // PATCH_SIZE   # 7 for 28×28 + patch_size=4
def mask_fn(n_p, device):
    return random_block_mask(n_patches=n_p, grid_size=grid_size, device=device)
```

`mask_fn` is called once per batch inside the train step; each image gets a fresh
context/target split. The notebook samples one target block per step (the paper uses four);
this is a tuning knob, not a correctness constraint.

### Custom pretrain step with EMA

```python
step_fn = jepa_train_step_factory(
    target_encoder=target_encoder, predictor=predictor,
    mask_fn=mask_fn, ema_momentum=EMA_MOMENTUM,
)
run = model.train(
    params=NNTrainParams(n_epochs=PRETRAIN_EPOCHS, train_loader=train_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=PRETRAIN_LR,
                            momentum=(0.9, 0.999), weight_decay=1e-4)),
    train_step_fn=step_fn,
)
```

The factory hides the bookkeeping: mask the batch, run the context encoder on context patches,
run the (frozen) target encoder on target patches, run the predictor, compute the prediction
loss against stopped-gradient targets, backprop into context encoder + predictor, then EMA the
target encoder.

### Frozen-encoder linear probe

```python
def _precompute_embeddings(encoder, loader, device):
    encoder.eval()
    embeds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            patch = encoder(x.float().to(device))   # (B, n_patches, d_model)
            embeds.append(patch.mean(dim=1).cpu())
            labels.append(y)
    return torch.cat(embeds), torch.cat(labels)

train_embeds, train_labels = _precompute_embeddings(encoder, train_loader, model.device)
probe_model = NNModel(net_params=NNParams(input_dim=D_MODEL, output_dim=10,
                                          hidden_dims=[], dropout_prob=0.0),
                      params=NNModelParams(net=Nets.FEED_FWD, device=DEVICE,
                                           loss=Losses.CROSS_ENTROPY))
```

Pre-computing the frozen embeddings once converts probe training into a fast pure-linear
optimization — five epochs on pre-computed features take seconds. Mean-pooling over the 49
patch embeddings produces a single \(d_{\text{model}}\)-dimensional vector per image, which is
the probe's input.

## 8.11.6 Results & analysis

On the recorded two-epoch pretrain + five-epoch probe run (seed 0, batch size 128):

| Metric | Value |
|---|---|
| ViT encoder params | 102,720 |
| JEPA predictor params | 30,368 |
| Patches per image | 49 (7×7 at `patch_size=4`) |
| JEPA pretrain iterations | 844 (2 epochs × 422 batches) |
| JEPA loss | 1.0416 → 0.3162 |
| Linear probe epochs | 5 |
| Linear probe final val accuracy | 74.43% |
| Random-baseline val accuracy | 10.00% (10 classes) |

The per-epoch probe trajectory:

| Epoch | Train CE | Val accuracy |
|---|---|---|
| 1 | 0.9482 | 70.62% |
| 2 | 0.7964 | 73.13% |
| 3 | 0.7619 | 73.37% |
| 4 | 0.7450 | 73.28% |
| 5 | 0.7315 | 74.43% |

Three observations:

1. **The JEPA prediction loss collapses sharply.** From 1.0416 to 0.3162 in two epochs — the
   predictor learns to map context-patch embeddings to target-patch embeddings. This validates
   the EMA + predictor + masking plumbing end-to-end; the loss is decreasing because the
   encoder is learning meaningful representations, not because of a degenerate collapse (which
   would also drop the loss, but to near-zero with a frozen constant predictor target).
2. **The frozen-encoder linear probe lands at ~74% — well above the 10% random baseline.** This
   is the SSL proof of life: the *frozen* pretrained encoder produces non-trivial features
   without ever seeing a Fashion-MNIST label. At two pretrain epochs this is far below the
   ~93%+ supervised Fashion-MNIST ceiling, but the gap is exactly the budget lever (longer
   pretrain closes it).
3. **The probe converges fast and cleanly.** Val accuracy jumps from 70.62% at epoch 1 to
   74.43% at epoch 5 with no overfitting drift — the frozen features are linearly separable
   enough that a single linear layer exhausts them in a handful of epochs.

The pedagogical headline: **I-JEPA replaces contrastive SSL with a predictive task in embedding
space** — no view augmentations, no negative samples, no collapse-prevention tricks beyond the
slow EMA. `nnx` ships the recipe as five composable primitives.

## 8.11.7 Pitfalls & edge cases

- **Heaviest Tier-A notebook (~90 s).** The ViT forward (49 patches × 64 \(d_{\text{model}}\) ×
  2 layers) is slow on CPU even at this scale. A convolutional encoder would be faster but the
  I-JEPA recipe is ViT-canonical; the cost is the price of recipe fidelity.
- **EMA momentum is stability-critical.** `0.996` means the target encoder takes roughly 250
  steps to meaningfully update. Lower values (0.99 or below) let the target chase the context
  encoder too quickly and the loss collapses to zero in trivial ways — the encoder-predictor
  pair finds a degenerate agreement. Do not lower this without watching the loss curve.
- **Single target block per step.** The I-JEPA paper uses four target blocks per image; this
  notebook's `mask_fn` samples one. A multi-block variant samples four blocks and aggregates
  the prediction losses — a clear extension, not a correctness constraint.
- **Fashion-MNIST, not CIFAR or ImageNet.** The substitution keeps the recipe CPU-feasible and
  avoids a new download. Switching to CIFAR-10 means swapping the `NNDataset(ds_class=...)` to
  `CIFAR10` and bumping `image_size=32, in_channels=3` — the rest of the recipe is unchanged.
- **Rebuild the train loader at per-batch granularity.** `NNDataset`'s default loader packs the
  whole Fashion-MNIST train set into one batch; the notebook rebuilds at `batch_size=128`.
  Same caveat as the MoE and diffusion tasks.
- **`Activations.RELU` on the placeholder is inert.** The `NNModel` shell uses a placeholder
  `FeedFwdNN` that is never executed, so the activation choice does not affect JEPA training.
  It exists only for `NNModelParams` validation.
- **Linear probe accuracy is a *lower bound* on representation quality.** Mean-pooling over
  patches discards spatial information; a more expressive probe (e.g. attention pooling or a
  fine-tuned head) would report higher accuracy from the same frozen features.

## 8.11.8 Extensions & references

- **Lengthen the pretrain.** 150+ epochs is typical for I-JEPA. The probe gap to the supervised
  ceiling narrows steadily with pretrain compute; the two-epoch budget here is a smoke test.
- **Multi-block masking.** Change `mask_fn` to sample four target blocks per image and
  aggregate the prediction losses — the paper's default.
- **Scale the ViT and the dataset.** I-JEPA-Huge on ImageNet at 224² RGB is the paper's
  production setting; same recipe, just bigger. Switching to CIFAR-10 is the smallest
  intermediate step.
- **Switch to V-JEPA.** The video variant masks spatiotemporal patches; same predictive
  recipe, different patch geometry.
- **References.** Assran et al., *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture* (CVPR 2023) — the I-JEPA recipe. Bardes et al.,
  *VICReg* and Grill et al., *BYOL* — for the contrastive / non-contrastive SSL context that
  I-JEPA is reacting to. He et al., *Masked Autoencoders* (CVPR 2022) — the pixel-space
  predictive counterpart. The `nnx.ViTNN` + `JEPAPredictor` + `jepa_train_step_factory` API is
  the in-repo surface for this recipe.
