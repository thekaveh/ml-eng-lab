# peft-mnist-to-fmnist-dora-vs-lora-pytorch

## 1. Task summary

- **Task:** Parameter-efficient fine-tuning (PEFT) — cross-task adaptation via LoRA and DoRA at the same rank.
- **Datasets:** MNIST (pretrain) → Fashion-MNIST (adapt). Both 28×28 grayscale, 10 classes — same `input_dim=784`, `output_dim=10` so no architecture surgery is needed.
- **Model:** `nnx.FeedFwdNN` (`Nets.FEED_FWD`), `Activations.RELU`, `hidden_dims=[128, 64]`.
- **Framework:** PyTorch (via [`nnx`](../nnx)).

## 2. Why this exists

LoRA (Hu et al., 2021) is the dominant adapter recipe: freeze the pretrained weights, augment each `Linear` `W ∈ ℝᵈˣᵏ` with a rank-`r` low-rank update `(α/r) BA` and train only `A`, `B`. **DoRA** (Liu et al., NVIDIA ICML 2024 Oral) refines LoRA with a magnitude/direction decomposition `W = m · V / ‖V‖_c` where `V = W₀ + (α/r) BA` and `m` is a trainable per-output-row magnitude vector initialized from `‖W₀‖_c`. At step 0 both keep `forward(x) == base(x)` exactly.

`nnx.apply_lora_to(net, "layers.*", r=8, alpha=16)` and `nnx.apply_dora_to(net, "layers.*", r=8, alpha=16)` wrap every matched `Linear` in the corresponding adapter, freeze the base, and return the count of wraps. `nnx.save_lora_weights` + `nnx.load_lora_weights` persist *only* the adapter tensors (the base stays where it is — typical adapter files are a few dozen KB).

This notebook runs both recipes on the same MNIST → Fashion-MNIST cross-task adaptation so the recipe-level trade-offs are directly readable.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — LoRA + DoRA recipes, dataset pair, libraries.
- §2 Environment & Setup — imports, hyperparameters (`LORA_RANK=8`, `LORA_ALPHA=16`, `PRETRAIN_EPOCHS=2`, `ADAPT_EPOCHS=1`), `nnx.set_seed(0)`.
- §3 Data — MNIST + Fashion-MNIST via `nnx.NNDataset`, same `(0.1307, 0.3081)` normalization so cross-task signal isn't muddied by per-dataset normalization.
- §4 Model — shared FFN architecture and PEFT contracts.
- §5 Training — pretrain on MNIST, snapshot state_dict, then three Fashion-MNIST adapt paths from the same pretrained base: full fine-tune control, LoRA r=8, DoRA r=8. Plus `save_lora_weights` + `load_lora_weights` round-trip into a fresh adapter from the same pretrained base.
- §6 Evaluation & Results — comparison table (trainable params + percentage of base + Fashion-MNIST val accuracy) for the four recipes (`full fine-tune`, `LoRA`, `DoRA`, `LoRA round-trip`); §6.2 discussion of budget-dependent trade-offs.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~25 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — MNIST + Fashion-MNIST + tensors.
- `nnx` (the submodule) — `FeedFwdNN`, `NNModel`, `NNDataset`, `apply_lora_to`, `apply_dora_to`, `save_lora_weights`, `load_lora_weights`, `LoRALinear`, `DoRALinear`.
- `prettytable` — comparison table.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Short training budget hides the PEFT win.** The recorded run uses `PRETRAIN_EPOCHS=2` + `ADAPT_EPOCHS=1` to stay Tier-A (~25 s on CPU). At that budget the full-fine-tune control beats LoRA / DoRA by a wide margin on Fashion-MNIST val accuracy. Real PEFT wins land on *well-converged* bases — extend `PRETRAIN_EPOCHS` to 10+ and `ADAPT_EPOCHS` to 5+ and the LoRA / DoRA accuracy closes most of the gap. The §6.2 prose owns this trade-off and points at it as the takeaway.
- **DoRA vs LoRA is too tight to distinguish at this budget.** At a long horizon DoRA typically beats LoRA by ~0.5–2 pp on cross-task benchmarks; at 1 epoch of Fashion-MNIST training the two are essentially tied. Not a defect — a known scale lever.
- **Same normalization across datasets.** We reuse MNIST's `(0.1307, 0.3081)` mean/std for Fashion-MNIST so the only cross-task variable is the label space. In production you'd use per-dataset stats.
- **`apply_*_to` requires at least one name-pattern.** Both `apply_lora_to(net)` and `apply_dora_to(net)` with no patterns raise `ValueError("at least one ...")`. The pattern `"layers.*"` matches every `Linear` inside `FeedFwdNN.layers`.
