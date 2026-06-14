# diffusion-mnist-ddpm-pytorch

## 1. Task summary

- **Task:** Generative modeling — train a tiny DDPM (Denoising Diffusion Probabilistic Model) on MNIST and sample new digits.
- **Dataset:** MNIST handwritten digits via `nnx.NNDataset`.
- **Model:** `nnx.DiffusionMLP(input_dim=784, hidden_dims=[256, 256], time_embed_dim=32)` — ~477 k parameters. Operates on flattened pixels (no U-Net).
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)) — exercises the megamerge diffusion stack end-to-end.

## 2. Why this exists

DDPM (Ho et al., 2020) is the foundational diffusion-model recipe. The `nnx` megamerge (thekaveh/NNx#29) ships the full stack: `NoiseSchedulers.LINEAR(T=...)` builds a `NoiseSchedule`; `DiffusionMLP(input_dim, hidden_dims, time_embed_dim)` is a denoiser that takes `(x_t, t)` and predicts ε; `diffusion_train_step_factory(schedule)` produces the noise-prediction `train_step_fn`; `sample(model, schedule, shape)` runs the reverse-diffusion loop. This notebook is the canonical in-repo demo of all four pieces working together on MNIST.

The model is *intentionally tiny* — a 3-layer MLP denoiser on flattened 784-D pixels, T=100 timesteps, 3 epochs on CPU. Generated samples are blurry and mode-mixed at this scale + budget; the point is the *pipeline* (schedule → train step → sampler), not the generation quality. §6.3 calls out the scale levers — swap `DiffusionMLP` for a U-Net and the same three calls produce a much better generator.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/diffusion-mnist-ddpm-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — DDPM recipe, the four `nnx` pieces, libraries.
- §2 Environment & Setup — imports, hyperparameters (`T=100`, `DENOISER_HIDDEN=[256, 256]`, `TIME_EMBED_DIM=32`, `N_EPOCHS=3`, `BATCH_SIZE=128`), `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on MNIST + custom `DataLoader(batch_size=128)` (NNDataset's default packs the whole 54k train set into one batch — too few noise-level samples per epoch for diffusion to learn).
- §4 Model — `NNModel` shell with a placeholder `FeedFwdNN` swapped for `DiffusionMLP` (`.net` substitution); noise schedule + train step contract.
- §5 Training — `diffusion_train_step_factory(schedule)` + `model.train(..., train_step_fn=step_fn)`. The recorded run does 1,266 iterations (3 epochs × 422 batches), noise-prediction loss 1.0102 → 0.9317.
- §6 Evaluation & Results — `sample(model, schedule, shape=(16, 784))` draws 16 samples via the reverse-diffusion loop, reshaped to 28×28 and rendered as a 4×4 grid; §6.3 discussion of capacity bottlenecks + scale levers.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~19 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — MNIST + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `DiffusionMLP`, `NoiseSchedulers`, `diffusion_train_step_factory`, `sample`, `NNModel`, `NNDataset`, `set_seed`.
- `matplotlib` — sample grid rendering.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Generated digits are blurry / mode-mixed.** Expected at this scale + budget (3 epochs × MLP denoiser on flat pixels × T=100). The §6.3 prose owns this; scaling levers are U-Net denoiser, larger T, more training, DDIM sampling.
- **`NNDataset` default batch_size is the whole train set** (54,000 for MNIST). For diffusion this gives ~1 iteration per epoch — far too few noise-level samples for the denoiser to learn. The notebook rebuilds `train_loader = DataLoader(ds.train_loader.dataset, batch_size=128, shuffle=True)` in §3.1 to fix this.
- **MLP on flattened pixels loses spatial structure.** A U-Net would let the denoiser exploit translation symmetries; the MLP doesn't and so over-fits the input distribution's global pixel statistics rather than the local stroke geometry.
- **`Normalize(mean=0.1307, std=0.3081)` shifts the pixel range.** Samples are un-normalized via `samples * DS_STD + DS_MEAN` in §6.1 before clamping to `[0, 1]` for display. If you'd skipped normalization, the diffusion math would be the same but the samples would be in `[0, 1]` directly.
