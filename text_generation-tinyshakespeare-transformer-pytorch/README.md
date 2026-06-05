# text_generation-tinyshakespeare-transformer-pytorch

## 1. Task summary

- **Task:** Autoregressive language modeling — train a tiny decoder-only transformer on a Shakespeare-style corpus, then sample text generations.
- **Dataset:** ~7 KB of Shakespeare embedded inline (lines from *Romeo & Juliet*, *Hamlet*, *Macbeth*, *Julius Caesar*, *As You Like It*), tiled 8× for training-loop size + BPE frequency.
- **Model:** `nnx.TransformerNN` via `Nets.TRANSFORMER` (decoder-only, RoPE positional, tied embeddings) — `n_layers=2`, `d_model=64`, `n_heads=4`, `max_seq_len=32`, ~114 k parameters.
- **Framework:** PyTorch (via [`nnx`](../nnx)) — exercises the megamerge transformer fork end-to-end.

## 2. Why this exists

The nnx megamerge (thekaveh/NNx#29) added a full decoder-only transformer fork: `TransformerNN` + `NNTransformerParams` + `Nets.TRANSFORMER` enum + `GenerativeNNModel.generate` + the `nnx.tokenizer` BPE trainer + the `nnx.generation` sampling stack (`RepetitionPenalty`, `TemperatureScaling`, `TopKFilter`, `TopPFilter`). This notebook is the first in-repo demo that walks every piece of that stack on a single self-contained corpus, on CPU, in under a minute. The result is *not* coherent Shakespeare — at 114 k params trained on 14 KB for 5 epochs, you get plausible-distribution gibberish — and that's the point. The notebook is a *correctness smoke test* + an executable reference for the transformer call chain. Real generation quality is a scale lever, discussed in §6.3.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — transformer fork, dataset, libraries.
- §2 Environment & Setup — imports, hyperparameters (`SEQ_LEN=32`, `D_MODEL=64`, `N_HEADS=4`, `N_LAYERS=2`, `N_EPOCHS=5`), `nnx.set_seed(0)`.
- §3 Data — embedded Shakespeare corpus tiled 8×, `train_bpe` trains a 256-vocab BPE tokenizer, the corpus is encoded into a single id stream and sliced into fixed-length 32-token windows with `y = x.roll(-1)` next-token targets.
- §4 Model — `NNTransformerParams` construction, parameter count, `GenerativeNNModel(net_params, params, tokenizer)`.
- §5 Training — custom `lm_train_step(ctx)` flattens (B, T, V) → (B*T, V) for `cross_entropy`; passed to `model.train(..., train_step_fn=lm_train_step)`.
- §6 Evaluation & Results — `model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=20, seed=42)` for three prompts; training loss trajectory; §6.3 discussion of scaling levers (bigger corpus, bigger model, longer training, real TinyStories via HF).

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~8 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch` — tensors + autograd.
- `nnx` (the submodule) — `TransformerNN`, `NNTransformerParams`, `GenerativeNNModel`, `train_bpe`, `NNTokenizerParams`, `TrainStepContext`, `NNEvaluationDataPoint`, `set_seed`.
- `matplotlib` — loss trajectory plot.

The `nnx.tokenizer` BPE trainer pulls in `tokenizers` (HuggingFace's Rust-backed BPE implementation) via nnx's `[lm]` extra — `requirements.txt` pins `-e ./nnx[lm]`, which resolves to `tokenizers>=0.20` + `datasets>=2.20` from the `[project.optional-dependencies] lm = [...]` block in `nnx/pyproject.toml`. The notebook itself doesn't import `tokenizers` directly.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Generations are not coherent.** At 114 k params + 14 KB corpus + 5 epochs the model overfits visibly on the embedded corpus but doesn't have nearly enough capacity / data for fluent text. The §6.3 prose owns this trade-off and points at the scaling levers.
- **Corpus is embedded, not downloaded.** The full Karpathy TinyShakespeare is ~1 MB — about 70× the embedded slice. Switching to it would mean a network download in CI, which we deliberately dodge after issue #3 (CI hangs on dataset downloads). The embedded form is also self-contained — anyone reading the notebook on nbviewer can see the corpus inline.
- **`drop_last=True` on the loader.** Combined with `BATCH_SIZE=4` + ~175 windows per epoch, this drops at most 3 windows; not a correctness concern. Without `CORPUS_REPEAT=8` the loader would yield 0 batches per epoch — the prior version of this notebook hit `IndexError: list index out of range` from an empty `idps` list at `model.train(...)`'s final aggregation step before we tiled the corpus.
- **Loss values are roughly per-batch sums, not per-token means.** The first-iter loss shows ~63 and falls to ~8 over 215 iterations — a clean monotonic decrease (the trajectory is the pedagogical signal; the absolute scale isn't directly comparable to the bits-per-character or perplexity numbers in the LM literature).
- **No validation loader.** This is autoregressive LM training on a tiny corpus — train loss is the only signal worth tracking, so `NNTrainParams.val_loader=None` and the notebook skips a val-loss curve. Real LM training would slice off a held-out chunk for perplexity tracking.
