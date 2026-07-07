# 8.19 Preference alignment — toy DPO

A comprehensive walk-through of `notebooks/preference_alignment-toy-dpo-pytorch/` — the in-repo
canonical demo of Direct Preference Optimization (Rafailov et al., 2023) on a tiny
`TransformerNN`. This page is the deep-dive companion to the task notebook: it states the problem,
builds the DPO loss math from first principles, dissects the reference-policy + contrastive-step
architecture, reads the code top to bottom, reports the measured chosen−rejected log-prob gap, and
lands the central contract — **DPO training must strictly increase the policy's chosen−rejected
log-prob gap relative to the reference.**

The notebook is **Tier-A** — CPU re-runs in about seven seconds and it is re-executed end-to-end
in CI on every pull request. The model is deliberately tiny (4 976 parameters, 16-dim residual
stream) and the preference corpus is 16 hand-written `(prompt, chosen, rejected)` triplets
(cheerful chosen, gloomy rejected) so the whole DPO recipe — BPE tokenizer, `NNPreferenceDataset`,
auto-freezing reference, `dpo_train_step_factory`, before/after gap measurement — runs on a laptop.
The recorded gap is large because the corpus is tiny and the policy overfits; the *contract* (gap
must increase) holds, which is the pedagogical point.

## 8.19.1 Problem & motivation

Aligning a language model to human preferences is the post-pretraining step that turns a
next-token predictor into a helpful assistant. The dominant historical recipe is RLHF (reward
model + PPO); Direct Preference Optimization (DPO) replaces that two-stage pipeline with a single
contrastive loss over `(prompt, chosen, rejected)` triplets, with no reward model and no RL.

This notebook exists for three reasons:

1. **First in-repo exercise of the `nnx` DPO stack.** The megamerge ships the full recipe:
   `NNPreferenceDataset` packages the triplets (prompt/chosen/rejected tokenization, padding,
   batching), `dpo_train_step_factory(ref_model, beta=, pad_token_id=)` produces the
   `train_step_fn` consumed by `policy.train(...)`, and the reference model is *automatically
   frozen* inside the factory (every parameter `requires_grad=False`, `.eval()` mode). This
   notebook walks every piece end-to-end on a single self-contained corpus, on CPU.
2. **Make the DPO contract concrete on real numbers.** The notebook measures the
   chosen−rejected log-prob gap before training, runs DPO, re-measures after, and asserts the gap
   strictly increased. On the recorded run the gap moves from `+2.1434` (before) to `+59.6521`
   (after) — a delta of `+57.5087`. The absolute magnitude reflects overfitting on 16 triplets;
   the sign and direction are the contract.
3. **Reusable seam for the §8.16 generative stack.** DPO rides on the same `GenerativeNNModel` +
   `TransformerNN` + `train_step_fn=` hook as the §8.16 language-modeling notebook. The only
   difference is the *step function*: §8.16 injects `lm_train_step` for next-token cross-entropy;
   this notebook injects the DPO factory's step for contrastive preference loss. A reader who
   understands both notebooks has the full generative-and-alignment skeleton.

The falsifiable hypothesis tested by the notebook is that even at toy scale — 16 triplets, 12
epochs, a 4 976-parameter transformer — DPO training strictly increases the policy's
chosen−rejected log-prob gap relative to the frozen reference. The notebook asserts this and
fails loudly if it ever inverts.

## 8.19.2 Concepts

| Concept | Where it shows up |
|---|---|
| Preference alignment | `(prompt, chosen, rejected)` triplets; learn to prefer chosen over rejected |
| Direct Preference Optimization (DPO) | Contrastive loss replacing reward-model + PPO |
| Reference vs policy model | `ref_model` frozen; `policy` trained; both start bit-identical |
| Chosen−rejected log-prob gap | The metric DPO optimizes; must strictly increase |
| Temperature β | `beta=0.1` trades off "stay close to ref" (low β) vs "move toward preference" (high β) |
| Auto-freezing | `dpo_train_step_factory` sets `requires_grad=False` + `.eval()` on the reference |
| `NNPreferenceDataset` | Packages triplets with prompt/response length bounds + padding |
| BPE tokenizer (tiny) | `train_bpe` on a 10-line corpus; effective vocab 52 (target 80 unreachable) |
| Decoder-only transformer (tiny) | `TransformerNN`, `d_model=16`, `n_layers=2`, `n_heads=2` |
| Implicit reward | DPO is the optimal policy under a Bradley-Terry reward derived from the data |

The `nnx` surface consumed: `GenerativeNNModel`, `NNTransformerParams`, `NNTokenizerParams`,
`NNModelParams`, `NNTrainParams`, `NNOptimParams`, `NNPreferenceDataset`,
`dpo_train_step_factory`, `train_bpe`, plus the enums `Devices`, `Losses`, `Nets`, `Optims` and
`nnx.set_seed`. The factory + dataset pair is the DPO-specific half; everything else is shared
with §8.16.

## 8.19.3 Mathematical formulation

DPO derives a maximum-likelihood objective for preference data from the RLHF setup, eliminating
the reward model. Let \(\pi_{\mathrm{ref}}\) be the frozen reference policy (the starting point)
and \(\pi_\theta\) the trainable policy. For a triplet \((x, y_w, y_l)\) — prompt \(x\), chosen
(winning) response \(y_w\), rejected (losing) response \(y_l\) — define the implicit reward as the
log-ratio of policy to reference, summed over response tokens:

\[
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}.
\]

Under a Bradley-Terry preference model, the probability that \(y_w\) is preferred over \(y_l\) is
the sigmoid of the reward difference:

\[
P(y_w \succ y_l \mid x) = \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right).
\]

Maximizing the log-likelihood of the observed preferences gives the DPO loss:

\[
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right].
\]

Two observations from the form:

1. **At initialization \(\pi_\theta = \pi_{\mathrm{ref}}\)** (the policy starts as a bit-identical
   copy), so both log-ratios are zero, the sigmoid argument is zero, and
   \(\mathcal{L}_{\mathrm{DPO}} = -\log \sigma(0) = \log 2 \approx 0.6931\). The recorded
   first-iteration loss is exactly `0.6931`, confirming the reference and policy start
   identical — a built-in sanity check.
2. **The loss pushes the policy to raise \(\log \pi_\theta(y_w \mid x)\) and lower
   \(\log \pi_\theta(y_l \mid x)\),** *both relative to the reference*. This is why the
   chosen−rejected log-prob gap,

\[
\mathrm{gap}(\theta) = \mathbb{E}\!\left[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)\right],
\]

is the metric the notebook measures before and after training. The DPO contract is that
\(\mathrm{gap}\) strictly increases from its pre-training value — equivalently, the policy becomes
*more* confident in chosen over rejected than the reference was.

The β knob trades off "stay close to the reference" (low β → small KL move, slow preference
update) vs "move aggressively toward the preference" (high β → large move, risk of degenerating).
`beta=0.1` is the recipe-paper default; production sweeps `β ∈ {0.01, 0.1, 0.5}` and picks by
held-out win-rate.

## 8.19.4 Architecture

Two identical transformer models — a frozen reference and a trainable policy — sharing the same
architecture and the same starting weights.

**Model (`Nets.TRANSFORMER`):** the same decoder-only transformer as §8.16, one notch smaller:

| Knob | Value | Role |
|---|---|---|
| `n_layers` | `2` | Two transformer blocks |
| `d_model` | `16` | Residual-stream width |
| `n_heads` | `2` | Two attention heads, \(d_k = 8\) |
| `max_seq_len` | `64` | Covers prompt + response (8 + 8 padded) |
| `ffn_mult` | `2` | FFN hidden width \(= 2 \times 16 = 32\) |
| `dropout_prob` | `0.0` | Disabled |
| `vocab_size` | `52` | Effective BPE vocab (target 80 unreachable on a 10-line corpus) |

Recorded parameter count: **4 976**.

The DPO-specific contract:

- **Reference:** `make_lm()` built after `nnx.set_seed(0)`; frozen inside
  `dpo_train_step_factory` (`requires_grad=False`, `.eval()`).
- **Policy:** `make_lm()` built after a *second* `nnx.set_seed(0)` so the init RNG matches the
  reference, then `policy.net.load_state_dict(ref_model.net.state_dict())` as a belt-and-suspenders
  guard so any init drift cannot violate the DPO assumption that policy starts == reference.
- **Loss:** the DPO contrastive loss above, applied via the factory's `train_step_fn`.
- **Optimizer:** `Optims.ADAM`, `max_lr=5e-3`, `momentum=(0.9, 0.999)`, `weight_decay=0.0`.
- **β:** `0.1`; `pad_token_id=1` (the `<pad>` id in the special-token list).
- **Device:** `Devices.CPU`; **Epochs:** `12`; **Batch size:** `4`; **Seed:** `0`.

The data: 16 `(prompt, chosen, rejected)` triplets (5 distinct tuples cycled, "cheerful chosen,
gloomy rejected" — e.g. `"the cat"` → chosen `"is happy and warm"`, rejected `"sat on the mat"`).
A 10-line tokenizer corpus trains the BPE tokenizer. `NNPreferenceDataset` tokenizes and pads each
triplet to `max_prompt_len=8` / `max_response_len=8`, batches 4-at-a-time → 4 batches/epoch × 12
epochs = **48 iterations**.

## 8.19.5 Code walkthrough

### Reference and policy construction

```python
def make_lm():
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size, output_dim=tokenizer.vocab_size,
        dropout_prob=0.0, vocab_size=tokenizer.vocab_size,
        n_layers=N_LAYERS, n_heads=N_HEADS,
        d_model=D_MODEL, ffn_mult=2, max_seq_len=MAX_SEQ_LEN,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=DEVICE, loss=Losses.CROSS_ENTROPY)
    return GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)

nnx.set_seed(0)
ref_model = make_lm()
nnx.set_seed(0)                                  # same init RNG for policy
policy = make_lm()
policy.net.load_state_dict(ref_model.net.state_dict())   # belt-and-suspenders: policy == ref
```

The seed reset + `load_state_dict` pair is the explicit guarantee that `policy` starts
bit-identical to `ref_model`. This matters because DPO's initialization-sanity check (loss = log 2
at step 0) only holds when the two models are identical; any drift would surface as a wrong
first-iteration loss and invalidate the §8.19.3 reasoning.

### Preference dataset

```python
ds = NNPreferenceDataset(
    prompts=PROMPTS, chosen=CHOSEN, rejected=REJECTED,
    tokenizer=tokenizer,
    max_prompt_len=MAX_PROMPT_LEN, max_response_len=MAX_RESPONSE_LEN,
    pad_token_id=PAD_TOKEN_ID,
    batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
    val_proportion=0.0, test_proportion=0.0,
    seed=0,
)
```

`NNPreferenceDataset` packages the three parallel lists into padded `(prompt, response)` tensor
pairs per triplet, with `prompt` and `response` truncated/padded to their respective length
bounds. The `batch_sizes` triple sets train/val/test batch sizes; with `val_proportion=0` and
`test_proportion=0`, all 16 triplets go to the train loader → 4 batches of 4.

### The DPO train step (factory)

```python
step_fn = dpo_train_step_factory(ref_model, beta=BETA, pad_token_id=PAD_TOKEN_ID)
```

This is the load-bearing line. The factory captures `ref_model`, freezes it
(`requires_grad=False` on every parameter, `.eval()` mode so dropout/batchnorm are inert), and
returns a `train_step_fn` that computes the DPO loss above for each batch. The freezing is
verified by the nnx test suite (bit-for-bit reference invariance after training); the notebook
never calls `requires_grad_` or `.eval()` itself — that is the factory's contract.

### Training

```python
run = policy.train(
    params=NNTrainParams(
        n_epochs=N_EPOCHS, train_loader=ds.train_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=LR,
                            momentum=(0.9, 0.999), weight_decay=0.0),
    ),
    train_step_fn=step_fn,
)
```

The `train_step_fn=step_fn` seam swaps the default classification step for the DPO contrastive
step — the same hook §8.16 uses for `lm_train_step`. Training runs 4 batches × 12 epochs = **48
iterations**; the resulting `NNRun` is checkpointed to `./runs/<run-id>`.

### The before/after gap measurement

```python
def _lp(seq):
    logits = net(seq)
    log_probs = torch.log_softmax(logits, dim=-1)
    resp_logits = log_probs[:, prompt_len - 1 : -1, :]
    resp_targets = seq[:, prompt_len:]
    return resp_logits.gather(dim=-1, index=resp_targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

# Before: policy == ref, so gap reflects the random init's incidental bias.
# After:  policy trained on the triplets.
# Assert: gap_after - gap_before > 0  (the DPO contract).
```

The `_lp` helper computes the token-summed response log-prob under a model — the raw ingredient
of both the DPO loss and the gap metric. The `[:, prompt_len - 1 : -1, :]` slice aligns the
log-prob of predicting response token \(t_i\) with the position that *generates* it (one step
before), and the `.gather(...)` picks out exactly the ground-truth response token's log-prob. The
gap is the mean of `(chosen_lp - rejected_lp)` across triplets; the notebook measures it on the
policy before and after training and asserts the delta is positive.

## 8.19.6 Results & analysis

On the recorded Tier-A run (`SMOKE_TEST=0`, 12 epochs, 48 iterations):

| Stage | Mean chosen−rejected log-prob gap |
|---|---|
| BEFORE DPO (policy == ref) | +2.1434 |
| AFTER DPO (policy trained) | +59.6521 |
| **delta (DPO contract)** | **+57.5087 (> 0 ✓)** |

DPO loss trajectory: `0.6931 → 0.0039`. The first-iteration loss `0.6931` equals `\log 2`, exactly
the theoretical DPO loss at init when policy == ref — a built-in sanity check that the two models
start identical. The final loss `0.0039` is essentially zero, meaning the policy has fully
separated chosen from rejected on the *train* triplets.

Three observations:

1. **The DPO contract holds.** The gap strictly increased (`+2.14 → +59.65`, delta `+57.51 > 0`).
   This is the load-bearing claim: DPO training moved the policy to prefer chosen over rejected
   *more than the reference did*. The notebook asserts this and would fail loudly if it ever
   inverted.
2. **The recorded gap is large because the corpus is tiny.** With 16 triplets and 12 epochs the
   policy overfits — chosen tokens get very high log-prob, rejected get very low. The `+59.65`
   after-gap reflects overfitting more than generalization; a real DPO run on thousands of
   triplets with held-out evaluation would show a much smaller, more honest gap. The contract
   (gap must increase) is the right invariant to test; the magnitude is not the right number to
   boast about.
3. **The before-gap is `+2.14`, not zero.** A random-init transformer does not produce a uniform
   distribution over tokens — its incidental biases already favor chosen tokens slightly on
   average. This is why DPO measures the *change* in gap relative to the reference, not the
   absolute gap. A before-gap of zero would be a coincidence; the `+2.14` baseline is the honest
   starting point the after-gap must beat.

The §6.3 prose owns the overfitting caveat and points at the scaling levers (real preference
data, β-sweep, held-out win-rate evaluation, DPO variants like IPO and KTO).

## 8.19.7 Pitfalls & edge cases

- **The recorded `+59.65` is the post-training gap, not the improvement.** The improvement is
  `+57.51`. The README's framing risks conflating the two; the gap *itself* moved from `+2.14` to
  `+59.65`, a delta of `+57.51`. Either reading confirms the DPO contract, but the magnitude to
  cite as "what DPO bought" is the delta.
- **The gap is large because of overfitting, not generalization.** With 16 train triplets and no
  held-out evaluation, the policy can memorize the specific chosen/rejected token sequences. Real
  DPO runs measure win-rate of the trained policy vs the reference on a held-out preference set;
  with 16 triplets there is no spare data to carve off, so the notebook measures only the train
  gap. Read the `+57.51` delta as "DPO works on the train signal," not "the policy generalized."
- **No held-out evaluation.** Real DPO measures win-rate on a held-out preference set. With 16
  train triplets the notebook does not bother carving off a val/test split; this is the right
  call at toy scale but the wrong call the moment the corpus grows.
- **`β=0.1` is the recipe-paper default, not a tuned value.** Production DPO sweeps
  `β ∈ {0.01, 0.1, 0.5}` and picks by held-out win-rate. Low β moves the policy slowly (safer);
  high β moves it aggressively (risk of degenerating). The notebook keeps the default for
  reproducibility.
- **The reference is frozen inside the factory, not in notebook code.** `dpo_train_step_factory`
  sets `requires_grad=False` and `.eval()` on `ref_model` — but you will not see those calls in
  the notebook. If you want to train another model afterwards, build a fresh `NNModel`; do not
  try to "un-freeze" the reference (its `.eval()` state and zeroed grad flags will leak).
- **The BPE target vocab (80) is unreachable on a 10-line corpus.** `train_bpe(vocab_size=80)`
  on 10 lines produces an effective vocab of 52 — there are not enough merge candidates. The
  transformer inherits `vocab_size=52` via `tokenizer.vocab_size`. Real LM tokenizers have vocab
  30 k–100 k; the tiny vocab here is a correctness smoke, not a quality claim.
- **No generation demo.** The notebook measures the log-prob gap, not actual generations from the
  trained policy. `policy.generate(prompt)` would give Shakespeare-style gibberish at this scale;
  the gap is the cleaner metric and the one that directly tests the DPO contract.

## 8.19.8 Extensions & references

- **Add a held-out win-rate evaluation.** Split the 16 triplets (or scale to 100+) into
  train/held-out; measure the trained policy's win-rate vs the reference on the held-out set via
  a judge (the reference itself, or an external reward model). This is the production-grade DPO
  metric and the one that exposes the overfitting caveat in §8.19.7.
- **Sweep β.** Run DPO at `β ∈ {0.01, 0.1, 0.5}` on the same triplets and report the gap
  trajectory + held-out win-rate for each. The β that maximizes held-out win-rate is the one to
  ship; the recipe default is a starting point, not a tuned value.
- **Try DPO variants: IPO, KTO.** Identity Preference Optimization (IPO) adds a regularization
  term to prevent the loss from collapsing to zero on small datasets; KTO (Kahneman-Tversky
  Optimization) drops the paired-triplet requirement and trains on unpaired preferred/non-preferred
  examples. Both are one-line swaps in the `train_step_fn` and exercise the same `train_step_fn=`
  seam.
- **Scale the model + corpus.** A real DPO run uses a 7B-parameter policy and thousands of
  triplets from a real preference dataset (Anthropic HH-RLHF, OpenAssistant, UltraFeedback). The
  gap shrinks to a few points but generalizes; the contract (gap > 0) still holds.
- **Reference reading.** Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model" (2023) for the derivation of the DPO loss from the RLHF objective;
  Ethayarajh et al., "IPO" (2023) for the regularization-on-small-data variant; the nnx examples
  directory for the reference `dpo_train_step_factory` implementation this notebook consumes.
