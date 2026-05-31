# preference_alignment-toy-dpo-pytorch

## 1. Task summary

- **Task:** Toy DPO (Direct Preference Optimization, Rafailov et al., 2023) preference alignment.
- **Dataset:** 16 hand-written `(prompt, chosen, rejected)` triplets — cheerful chosen, gloomy rejected.
- **Model:** Tiny `TransformerNN` (`d_model=16`, `n_layers=2`, `n_heads=2`, `max_seq_len=64`) — both `ref_model` (frozen) and `policy` (trained).
- **Framework:** PyTorch (via [`nnx`](../nnx)) — exercises the megamerge DPO + preference-dataset stack end-to-end.

## 2. Why this exists

DPO is the dominant alternative to RLHF for aligning a language model to human preferences. It replaces the reward-model + PPO pipeline with a single contrastive loss over `(prompt, chosen, rejected)` triplets:

```
L_DPO = -E[log σ( β·(log π_θ(chosen|prompt) - log π_ref(chosen|prompt))
              - β·(log π_θ(rejected|prompt) - log π_ref(rejected|prompt)) )]
```

The `nnx` megamerge ships the recipe: `NNPreferenceDataset` packages the triplets, `dpo_train_step_factory(ref_model, beta=0.1)` produces the `train_step_fn`. The reference model is *automatically frozen* at factory construction (every parameter `requires_grad=False`, `.eval()` mode) — the nnx test suite enforces bit-for-bit reference invariance after training.

This notebook is the canonical in-repo demo end-to-end. The model + corpus are tiny so the whole thing runs Tier-A on CPU.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/preference_alignment-toy-dpo-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — DPO loss + recipe, dataset, libraries.
- §2 Environment & Setup — imports, hyperparameters (`BETA=0.1`, `N_PAIRS=16`, `N_EPOCHS=12`, `MAX_PROMPT_LEN=8`, `MAX_RESPONSE_LEN=8`), `nnx.set_seed(0)`.
- §3 Data — BPE tokenizer trained on a 10-line corpus, 16 `(prompt, chosen, rejected)` triplets, `NNPreferenceDataset` wrapping.
- §4 Model — copy-the-reference policy + DPO contract (auto-freezing + `eval()` mode).
- §5 Training — measure chosen−rejected log-prob gap **before** training; train via `policy.train(..., train_step_fn=dpo_train_step_factory(ref_model, beta=0.1))`; re-measure gap **after**; assert the gap strictly increased.
- §6 Evaluation & Results — before/after table; DPO loss trajectory; §6.3 discussion of scaling levers (real preference data, β-sweep, DPO variants like IPO and KTO).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~7 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (the submodule) — `GenerativeNNModel`, `TransformerNN`, `NNTransformerParams`, `NNTokenizerParams`, `train_bpe`, `NNPreferenceDataset`, `dpo_train_step_factory`, `set_seed`.
- `matplotlib` — DPO loss trajectory plot.
- `prettytable` — before/after gap table.

All in the root `requirements.txt` + `torch-requirements.txt`. The `tokenizers` package (HuggingFace's Rust-backed BPE) is pulled in transitively by nnx's tokenizer module.

## 6. Known issues

- **Recorded gap is large because the train set is tiny.** With only 16 triplets and 12 epochs, the policy overfits — chosen tokens get very high log-prob, rejected get very low. The recorded "gap" of +59.65 is impressive-looking but reflects overfitting more than generalization. The DPO *contract* (gap must increase) still holds, which is the pedagogical point.
- **No held-out evaluation.** Real DPO runs measure win-rate of the trained policy vs the reference on a held-out preference set. With 16 train triplets we don't bother carving off a val/test split.
- **`β=0.1` is the recipe-paper default.** Production DPO sweeps `β ∈ {0.01, 0.1, 0.5}` and picks by held-out win-rate.
- **Tokenizer corpus is tiny** (10 lines, vocab 80). The BPE learns only ~20-30 useful merges; real LM tokenizers have vocab 30k–100k.
- **No generation demo.** We measure the log-prob gap, not actual generations from the trained policy. `policy.generate(prompt)` would give Shakespeare-style gibberish at this scale; the gap is the cleaner metric.
- **Reference model is in `.eval()` mode after factory construction**. If you want to train another model afterwards, build a fresh `NNModel` — don't try to "un-freeze" the reference.
