# moe-fmnist-mixture-of-experts-pytorch

## 1. Task summary

- **Task:** Mixture-of-Experts (MoE) classification with sparse expert routing + load-balancing aux loss.
- **Dataset:** Fashion-MNIST (10 apparel classes, 28×28 grayscale) via `nnx.NNDataset`.
- **Model:** `FeedFwdNN` subclass (`MoEClassifier`) whose first hidden layer is `nnx.MoELinear(in=784, out=128, num_experts=4, top_k=2)`. ~406k total params (router 3k + 4 experts × ~100k + classifier head 1.3k).
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)).

## 2. Why this exists

An MoE layer replaces a single `nn.Linear` with `num_experts` parallel linears + a learned router that picks `top_k` of them per token. Total parameter budget grows linearly with `num_experts`; per-token FLOPs stay roughly constant (only `top_k` experts run). The bet: experts can *specialize* on different input subdistributions.

The hard part is **load balancing**: without a penalty, the router collapses onto one or two experts. `nnx.moe_train_step_factory(aux_loss_weight=0.05)` augments the supervised step with the Switch-Transformer aux loss summed across every `MoELinear` layer. The notebook tracks `moe_layer.last_aux_loss` before and after training to verify the penalty is doing its job.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — MoE intuition, load-balancing, dataset, libraries.
- §2 Environment & Setup — imports, hyperparameters (`NUM_EXPERTS=4`, `TOP_K=2`, `AUX_LOSS_WEIGHT=0.05`, `N_EPOCHS=2`), `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on Fashion-MNIST + custom `DataLoader(batch_size=128)` (NNDataset's default packs the whole train set into one batch).
- §4 Model — `MoEClassifier` subclasses `FeedFwdNN` and swaps its first hidden Linear for `MoELinear`; parameter breakdown table; contracts.
- §5 Training — pre-training aux-loss snapshot on a probe batch (640 samples); train via `moe_train_step_factory(aux_loss_weight=0.05)`.
- §6 Evaluation & Results — aux-loss before/after table; expert-utilization bar chart on the probe (showing the fraction of tokens routed to each expert via the router's top-1); §6.3 discussion of why uniform isn't the goal — *useful specialization* can show up as 30/30/20/20.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~18 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — Fashion-MNIST + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `MoELinear`, `moe_train_step_factory`, `NNModel`, `NNDataset`, `FeedFwdNN`, `set_seed`.
- `matplotlib` — expert-utilization bar chart.
- `prettytable` — parameter breakdown + aux-loss tables.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Aux loss decreases but doesn't hit 1.0** at this budget. Recorded run: 1.2794 → 1.2218 (toward the 1.0 floor at uniform routing). Longer training + a larger `aux_loss_weight` push it closer to 1.0 at the cost of supervised-signal quality.
- **`NNDataset` default batch_size** is the whole train set (54k for Fashion-MNIST). Same caveat as the diffusion + N1 transformer tasks — we rebuild the train loader at `batch_size=128` in §3.1.
- **Expert utilization is data-dependent.** On Fashion-MNIST's 10 visually-similar apparel classes, even a well-trained router lands closer to 30/30/20/20 than perfect 25/25/25/25. Perfect uniform routing is *not* the goal — specialization is.
- **Single MoE layer.** Real MoE Transformers stack many MoE layers (one every other transformer block, typically). This notebook uses a single `MoELinear` to keep the demo's load-balancing story unambiguous. Adding more layers would amplify the aux-loss signal (summed across all `MoELinear` instances).
- **No comparison vs dense baseline.** The pedagogical point is the *MoE recipe and its load-balancing knob*, not "MoE beats dense on Fashion-MNIST" (it usually doesn't at this scale — MoE wins show up on harder problems where genuine specialization is possible).
