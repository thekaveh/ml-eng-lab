# self_supervised-fmnist-jepa-pytorch

## 1. Task summary

- **Task:** Self-supervised representation learning (I-JEPA — Image Joint-Embedding Predictive Architecture) + frozen-encoder linear probe.
- **Dataset:** Fashion-MNIST (28×28 grayscale, 10 apparel classes) via `nnx.NNDataset`.
- **Model:** `nnx.ViTNN` (`image_size=28`, `patch_size=4`, `in_channels=1`, `d_model=64`, `n_layers=2`, `n_heads=4`) — ~103k params; `nnx.JEPAPredictor` — ~30k params.
- **Framework:** PyTorch (via [`nnx`](../nnx)) — exercises the megamerge JEPA + ViT stack end-to-end.

## 2. Why this exists

I-JEPA (Assran et al., 2023) is Yann LeCun's framework for self-supervised representation learning. Unlike SimCLR / BYOL (contrastive — compare augmented views), JEPA is **predictive in embedding space**: given a context block of patches, predict the embeddings of a held-out target block. No view augmentations, no negative samples, no collapse-prevention tricks beyond the slow-EMA target encoder.

The `nnx` megamerge ships the full I-JEPA stack: `ViTNN`, `build_target_encoder` (deepcopy + freeze), `JEPAPredictor`, `random_block_mask`, `jepa_train_step_factory`. This notebook is the canonical in-repo demo end-to-end.

**Dataset substitution**: The original I-JEPA paper uses ImageNet (224² RGB); the nnx example uses CIFAR-10 (32² RGB). We use Fashion-MNIST (28² grayscale) because (a) it's already in the collection (no new download), (b) it's CPU-feasible at the Tier-A budget, and (c) the plumbing is identical — only `image_size`, `in_channels`, `patch_size` change.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/self_supervised-fmnist-jepa-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — I-JEPA recipe, Fashion-MNIST vs CIFAR substitution, libraries.
- §2 Environment & Setup — imports, hyperparameters (`PATCH_SIZE=4` → 49 patches/image, `D_MODEL=64`, `N_LAYERS=2`, `N_HEADS=4`, `EMA_MOMENTUM=0.996`), `nnx.set_seed(0)`.
- §3 Data — Fashion-MNIST via `NNDataset` + custom `DataLoader(batch_size=128)` (same NNDataset-default-batch workaround as diffusion + MoE + transformer tasks).
- §4 Model — `NNModel` shell with placeholder `FeedFwdNN`, swapped for `ViTNN`; `build_target_encoder` for the frozen EMA target; `JEPAPredictor` attached to `model.net` so the optimizer picks up its params jointly; `random_block_mask` mask sampler.
- §5 Training — Phase 1: JEPA pretrain via `jepa_train_step_factory(target_encoder, predictor, mask_fn, ema_momentum=0.996)`. Phase 2: build a `LinearProbe(encoder, embed_dim, num_classes)` — freezes the encoder, trains only a `Linear(d_model → 10)` head on top of mean-pooled patch embeddings.
- §6 Evaluation & Results — summary table + JEPA pretrain loss curve + linear-probe val-accuracy curve; §6.3 discussion of scaling levers (bigger ViT + ImageNet, multi-block mask, longer pretrain, V-JEPA video variant).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (heavier — ~90 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell; `SMOKE_TEST=1` drops both pretrain + probe to 1 epoch each.

## 5. Dependencies

- `torch`, `torchvision` — Fashion-MNIST + tensors.
- `nnx` (the submodule) — `ViTNN`, `JEPAPredictor`, `build_target_encoder`, `jepa_train_step_factory`, `random_block_mask`, `NNModel`, `NNDataset`, `set_seed`.
- `matplotlib` — pretrain loss + probe accuracy curves.
- `prettytable` — summary table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **~90 s runtime** is the heaviest Tier-A notebook in the collection (next-heaviest is model_surgery at ~45 s). The ViT forward (49 patches × 64 d_model × 2 layers) is slow on CPU even at this scale; a U-Net or convolutional encoder would be faster but the I-JEPA recipe is ViT-canonical.
- **Linear probe val accuracy ~74%** at this pretrain budget — well above the 10% random baseline (the SSL proof-of-life) but well below the ~93%+ Fashion-MNIST supervised baseline. Real I-JEPA needs 100+ pretrain epochs to close the gap; here we stay Tier-A.
- **Fashion-MNIST, not CIFAR**: see §1 for the rationale. Switching to CIFAR-10 means swapping the `NNDataset(ds_class=thv.datasets.FashionMNIST, ...)` to `CIFAR10` and bumping `image_size=32, in_channels=3` — the rest of the recipe is unchanged.
- **Single mask block per step.** The I-JEPA paper uses 4 target blocks per image. Our `mask_fn` returns 1; a multi-block variant would sample 4 blocks and aggregate the prediction losses. Future hyperparameter tuning.
- **`Activations.RELU` constraint on the placeholder.** The `NNModel` shell uses a placeholder `FeedFwdNN` (never executed), so the activation choice doesn't matter for the actual JEPA training — we use RELU just out of habit / consistency with the rest of the collection.
