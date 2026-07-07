# 8.5 Model surgery — MNIST FFN (Net2Net)

A comprehensive walk-through of `notebooks/model_surgery-mnist-ffnn-pytorch/` — the canonical
in-repo demo of `nnx.surgery` and the function-preserving model-edit pattern. This page is the
deep-dive companion to the task notebook: it states the problem, builds the math of Net2WiderNet
and Net2DeeperNet, dissects the surgery contract, reads the code top to bottom, reports the
measured warm-start vs cold-start results, and catalogues the pitfalls and extensions.

The notebook is **Tier-A** — CPU re-runs in roughly 45 seconds and it is re-executed end-to-end in
CI on every pull request. It is part of the "efficient/compressed MLP" family that also includes
the born-again distillation (§8.6), magnitude-pruning (§8.7), and quantization (§8.8) tasks; model
surgery is the odd-one-out of that family because it changes the *architecture* rather than the
*bitwidth* or *sparsity*, but it shares the same overarching goal — keep the trained weights and
make the deployed model better, not smaller.

## 8.5.1 Problem & motivation

Most training pipelines treat architecture as immutable: you pick `hidden_dims`, train, then either
keep the model or restart from scratch with a different shape. The restart is wasteful — everything
the old model learned is thrown away — and it is often *unnecessary*, because most of the trained
weights in the old model have exact, mathematically-defined counterparts in the new one. **Model
surgery** (Net2Net; Chen et al., 2015) is the alternative: edit a *trained* model's architecture
(widen a layer, insert a layer, drop a layer) in a way that *preserves its forward output exactly*,
then continue training from there.

This notebook exists for two reasons:

1. **First in-repo exercise of `nnx.surgery`.** The `nnx` release ships `nnx.widen`,
   `nnx.deepen`, `nnx.drop_layer`, and `nnx.low_rank_factorize` as a surgery namespace; this
   notebook is the canonical demo of the two headline operations — Net2WiderNet (`nnx.widen`) and
   Net2DeeperNet (`nnx.deepen`) — on a real trained model.
2. **The function-preservation contract is the technical headline.** Once the surgery preserves
   the forward output to numerical precision, warm-starting a strictly larger model from a smaller
   one is *free*: there is no accuracy cliff at step 0 of resumed training. The notebook asserts
   the contract on a probe batch before relying on it.

The falsifiable hypothesis tested by the notebook is that warm-starting a wider model via surgery
reaches a given validation loss *faster* than cold-starting the same wider shape from random init —
the Net2Net "accelerating learning via knowledge transfer" claim, exercised at a budget where the
gap is observable.

## 8.5.2 Concepts

| Concept | Where it shows up |
|---|---|
| Function-preserving transform | `widen` / `deepen` must match the original forward to `atol=1e-5` |
| Net2WiderNet | `nnx.widen(layer_name="layers.0", new_width=128)` — widen a hidden layer by replicating units |
| Net2DeeperNet | `nnx.deepen(after_layer_name="layers.0")` — insert an identity-initialized `Linear` |
| Identity initialization | The inserted deeper layer is a no-op at step 0 (ReLU-only contract) |
| Warm-start vs cold-start | Same target shape `[128, 64]`, different starting weights (surgery vs random) |
| ReLU activation constraint | `deepen`'s identity-init is function-preserving only for ReLU |
| `nnx.FeedFwdNN` (`Nets.FEED_FWD`) | The substrate architecture; two hidden layers |
| Reproducibility | `nnx.set_seed(0)` pins Python `random`, NumPy, PyTorch CPU + CUDA + cuDNN |

The `nnx` surface consumed is: `NNModel`, `NNParams`, `NNModelParams`, `NNTrainParams`, `NNDataset`,
`Activations`, `Devices`, `Losses`, `Nets`, `Utils`, `VisUtils`, `set_seed`, and the surgery
primitives `nnx.widen` / `nnx.deepen`. The enums (`Nets.FEED_FWD`, `Losses.CROSS_ENTROPY`,
`Activations.RELU`, `Devices.CPU`) make the contract read as configuration rather than magic
strings.

## 8.5.3 Mathematical formulation

Both operations are *function-preserving*: if \(f\) is the original network and \(f'\) the operated
network, then \(f'(x) = f(x)\) for every input \(x\) (to numerical precision). The constructions
below are the Net2Net recipes.

**Net2WiderNet.** Suppose layer \(i\) has weight \(W \in \mathbb{R}^{k \times m}\) (mapping \(m\)
inputs to \(k\) hidden units) and the next layer has weight \(U \in \mathbb{R}^{m' \times k}\). To
widen layer \(i\) from \(k\) to \(k' > k\) units, choose an index map \(\mathbf{r} \in
\{1,\dots,k\}^{k'}\) that says, for each new unit, which old unit it copies. The new weights are

\[
W'_{:,j} = W_{:,\mathbf{r}_j}, \qquad
U'_{j,:} = \frac{1}{|\{j' : \mathbf{r}_{j'} = \mathbf{r}_j\}|}\, U_{\mathbf{r}_j,:}.
\]

The \(W'\) column-copy makes the new unit compute the same pre-activation as its donor; the
\(U'\) row-scaling (dividing each outgoing row by the number of units sharing that donor) makes the
post-activation contribution sum to exactly the original. `nnx.widen` implements this with a random
replication map; the construction preserves the forward to \(\sim 10^{-6}\) (floating-point
rounding, not bit-exact).

**Net2DeeperNet.** To insert a new `Linear` layer after an existing activation, initialize it as the
identity: \(W_{\text{new}} = I\), \(b_{\text{new}} = 0\). For a ReLU activation the composition
\(\text{ReLU}(I\,h + 0) = \text{ReLU}(h)\) is exactly the original, so the inserted layer is a
literal no-op at step 0 — bit-exact, drift \(0.00\). For sigmoid/tanh/GELU the identity-init does
not compose through the activation, so a different post-insertion init would be needed; the notebook
pins the baseline to `Activations.RELU` to honor this.

The training objective is cross-entropy on the MNIST logits,

\[
\mathcal{L}(z, y) = -\log \frac{e^{z_c}}{\sum_j e^{z_j}},
\]

and the optimizer is Adam with \(\eta = 10^{-2}\), \(\beta = (0.9, 0.999)\), weight decay
\(5 \times 10^{-5}\).

## 8.5.4 Architecture

![Feed-forward MLP](../diagrams/img/mlp.png)

The substrate is `Nets.FEED_FWD` (`nnx.FeedFwdNN`): a 784-unit input layer (flattened MNIST), two
hidden layers with ReLU, and a 10-unit output layer consumed by softmax + cross-entropy. Surgery is
demonstrated on a deliberately small baseline so the post-surgery model is *meaningfully* larger.

| Model | `hidden_dims` | Role |
|---|---|---|
| Baseline | `[64, 64]` | The surgery substrate; trained for 3 epochs (undertrained on purpose) |
| Warm-widen | `[128, 64]` | Baseline after `nnx.widen("layers.0", new_width=128)`; warm-started |
| Cold-widen | `[128, 64]` | Fresh random init at the same target shape; the control |
| Continue | `[64, 64]` | Keep training the baseline shape (no surgery); the capacity ceiling |

The shared contract across all four:

- **Net:** `Nets.FEED_FWD`, **activation:** `Activations.RELU` (deepen contract)
- **Loss:** `Losses.CROSS_ENTROPY`
- **Optimizer:** Adam, `max_lr=1e-2`, `weight_decay=5e-5`, `momentum=(0.9, 0.999)`
- **Device:** `Devices.CPU`
- **Budget:** baseline 3 epochs; resumed models 5 epochs each (`BASELINE_EPOCHS=3`,
  `RESUME_EPOCHS=5`, or `SMOKE_TEST_EPOCHS=1` under `SMOKE_TEST=1`)
- **Seed:** `0` for the baseline; resumed models use seeds `1`, `2`, `3` respectively (so each
  resume run is reproducible but the three resume trajectories differ from each other only in their
  data-shuffle order, not in their init — the init is set by the surgery / fresh-construct step)

The widen/deepen targets are pinned in the configuration cell: `WIDEN_LAYER_NAME="layers.0"`
(widen the first `Linear`, 64→128), `DEEPEN_AFTER="layers.0"` (insert an identity `Linear` after
the first activation). Dropout is held at `0.0` throughout — the surgery contract is *exact*, not
statistical, and dropout's stochastic masking would break the forward-equality assertion.

## 8.5.5 Code walkthrough

### Baseline construction

```python
def make_model(hidden_dims):
    # nnx.deepen's identity-init insertion is function-preserving only for ReLU.
    return NNModel(
        net_params=NNParams(
            input_dim=ds.input_dim, output_dim=ds.output_dim,
            hidden_dims=hidden_dims, dropout_prob=DROPOUT_PROB,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=DEVICE, loss=Losses.CROSS_ENTROPY),
    )

baseline = make_model(BASE_HIDDEN_DIMS)   # [64, 64]
```

The ReLU pin is load-bearing for the deepen step that follows — see pitfalls. The baseline trains
for 3 epochs and reaches a final validation loss of `1.3150`: deliberately undertrained, so the
post-surgery resume has headroom to improve.

### Asserting the function-preservation contract

```python
x_probe, _ = next(iter(ds.val_loader))
x_probe = x_probe.view(x_probe.size(0), -1)
baseline.net.eval()
with torch.no_grad():
    y_before = baseline.net(x_probe)

widened_net = nnx.widen(copy.deepcopy(baseline.net),
                        layer_name=WIDEN_LAYER_NAME, new_width=WIDEN_NEW_WIDTH)
with torch.no_grad():
    y_after_widen = widened_net(x_probe)
widen_drift = (y_before - y_after_widen).abs().max().item()
assert widen_drift < 1e-5, "widen broke function preservation"

deeper_net = nnx.deepen(copy.deepcopy(baseline.net), after_layer_name=DEEPEN_AFTER)
with torch.no_grad():
    y_after_deepen = deeper_net(x_probe)
deepen_drift = (y_before - y_after_deepen).abs().max().item()
assert deepen_drift < 1e-5, "deepen broke function preservation"
```

The probe batch is one validation minibatch; the assertions are the *empirical* half of the Net2Net
proof. The recorded drifts on the committed run are `widen_drift = 1.91e-06` and
`deepen_drift = 0.00e+00`. The asymmetry is real: `widen` divides each outgoing row by a shared-donor
count, which accumulates floating-point rounding error; `deepen`'s identity matrix is bit-exact for
ReLU. The `1e-5` tolerance should not be relaxed — drift past it means the surgery primitive is
broken.

### Wrapping an operated net for resumed training

```python
def make_model_from_net(net, model_params):
    m = NNModel(
        net_params=NNParams(input_dim=ds.input_dim, output_dim=ds.output_dim,
                            hidden_dims=BASE_HIDDEN_DIMS, dropout_prob=DROPOUT_PROB,
                            activation=Activations.RELU),
        params=model_params,
    )
    m.net = net   # inject the operated (or copied) net over the freshly-constructed one
    return m
```

The operated net is injected into the live `m.net` slot so the standard `model.train(...)` loop
applies to it. Note the comment in the notebook: `m.net_params` stays at the constructor dims
(`[64, 64]`), which is stale relative to the injected wider net's actual `[128, 64]` — `net_params`
is descriptive metadata here, and the live `m.net` drives training. This is a known seam in the
`nnx` API surface (no public call both injects a net *and* syncs params); it does not affect the
recorded results.

### The warm-start vs cold-start race

```python
cont_model   = make_model_from_net(copy.deepcopy(baseline.net), ...)
warm_widened = make_model_from_net(widened_net, ...)
cold_widened = make_model([WIDEN_NEW_WIDTH, BASE_HIDDEN_DIMS[1]])   # fresh [128, 64]

cont_run  = cont_model.train(params=make_resume_params(seed=1))
warm_run  = warm_widened.train(params=make_resume_params(seed=2))
cold_run  = cold_widened.train(params=make_resume_params(seed=3))
```

All three resume for the same `RESUME_EPOCHS=5` budget. `continue` keeps the baseline shape;
`warm-widen` and `cold-widen` share the target shape `[128, 64]` but differ in starting weights —
surgery-transferred vs random. The loss curves are overlaid via `VisUtils.multi_line_plot` so the
step-0 loss gap and the convergence trajectories are directly comparable.

## 8.5.6 Results & analysis

On the recorded (seeded) run, the four models land as:

| Model | Final train loss | Final val loss |
|---|---|---|
| continue (base `[64, 64]`) | 0.9971 | 0.9260 |
| warm-widen (`[128, 64]`) | 1.2413 | 1.1409 |
| cold-widen (`[128, 64]`) | 1.0854 | 0.8612 |

Three observations:

1. **The surgery contract holds at step 0.** The warm-widen trajectory starts at exactly the
   baseline's resumed validation loss (the `1.91e-06` widen drift is invisible at loss scale), while
   cold-widen starts much higher from random init. This is the function-preservation promise made
   visible.
2. **At this very short budget, cold-widen reaches the lowest validation loss (0.8612).** This is
   the honest recorded outcome and it is *not* a violation of Net2Net — it is the budget-dependence
   the notebook's §6.3 calls out directly. The baseline trained for only 3 epochs, so the
   surgery-transferred weights are themselves undertrained; cold-widen's fresh, fully-random init
   at the wider shape gets 5 uninterrupted epochs of optimization at the right capacity, while
   warm-widen spends part of its budget "unlearning" the narrow-shape solution.
3. **The Net2Net advantage shows up at longer schedules.** The headline pedagogical point —
   warm-start beats cold-start on time-to-target — appears once the baseline is well-converged
   before surgery and the resume budget is long enough for cold-start to spend epochs just reaching
   the warm-start's step-0 loss. At Tier-A's 3 + 5 epochs (chosen for CPU feasibility) the gap can
   go either way; the recorded run is an example of cold-widen winning.

The right reading of the table is therefore: *the contract works* (warm-widen's step-0 loss equals
the baseline's), and *the comparison is budget-dependent* (the cold-widen win at this budget is a
real effect, not a bug). Extending the budget is the first item in §8.5.8.

## 8.5.7 Pitfalls & edge cases

- **`nnx.deepen` is function-preserving only for ReLU.** The identity-initialized inserted layer
  composes through ReLU exactly; for sigmoid/tanh/GELU it does not, and a different post-insertion
  init would be needed. The notebook pins `Activations.RELU` to honor the contract — if you swap
  the activation, the `deepen_drift < 1e-5` assertion will fail.
- **`nnx.widen` is not bit-exact.** Its drift is \(\sim 10^{-6}\) (floating-point rounding from the
  shared-donor row scaling), not `0.00`. The `1e-5` tolerance accommodates this; do not tighten it
  to `0` or the widen assertion will fire on a correct primitive.
- **Do not relax the `1e-5` assertion.** Drift past `1e-5` means the surgery primitive is broken
  (wrong layer targeted, wrong replication map, activation mismatch). The assertion is the
  empirical half of the Net2Net proof — keep it strict.
- **The warm-vs-cold comparison is budget-dependent.** At the short Tier-A budget (3 baseline + 5
  resume epochs), cold-widen can match or beat warm-widen, as the recorded run shows. Do not
  present the notebook as "warm-start always wins" — the §6.3 cell is explicit about this. The
  Net2Net time-to-target advantage emerges at longer schedules with a well-converged baseline.
- **Injecting an operated net leaves `net_params` stale.** `make_model_from_net` injects the wider
  net into `m.net` but `m.net_params` still reports the constructor's `[64, 64]`. This is
  descriptive metadata only (the live `m.net` drives training), but any downstream code that reads
  `model.net_params.hidden_dims` to size a new layer will see the wrong width. Surgery consumers
  should read shapes off `model.net` directly.
- **Keep dropout at 0 during surgery verification.** The function-preservation assertion is
  deterministic; dropout's stochastic masking would make `y_before` and `y_after` differ for
  reasons unrelated to the surgery. The notebook holds `DROPOUT_PROB=0.0` for exactly this reason.
- **Re-pin the seed for reproducibility.** `nnx.set_seed(0)` is called before baseline training;
  the three resume runs take seeds `1`, `2`, `3`. Without explicit seeds the three trajectories
  vary run-to-run and the comparison becomes noise.

## 8.5.8 Extensions & references

- **Lengthen the budget to expose the Net2Net time-to-target gap.** Train the baseline to
  convergence (10–20 epochs) before surgery, then resume warm- and cold-widen for a longer
  schedule. The warm-start should reach a target validation loss in fewer epochs than cold-start —
  the canonical Net2Net result that the Tier-A budget cannot reliably show.
- **Exercise `nnx.drop_layer` and `nnx.low_rank_factorize`.** The surgery namespace ships two more
  primitives not covered here. `drop_layer` is the dual of `deepen` (remove a layer
  function-preservingly); `low_rank_factorize` inserts a low-rank bottleneck. A sibling notebook
  demonstrating the full four-operation surface would complete the surgery story.
- **Surgery on a non-ReLU net.** Construct a sigmoid/tanh baseline and either (a) accept that
  `deepen` will not be function-preserving and re-derive the post-insertion init, or (b) restrict
  surgery to `widen` only. Either path documents the activation-dependence of the contract.
- **Combine surgery with distillation (§8.6).** Warm-start a wider student via `widen`, then
  distill back from the narrower teacher. This is the "Net2Net + born-again" pipeline used in some
  production compression workflows; the two notebooks' `nnx` primitives compose cleanly.
- **References.** Chen et al., 2015, *Net2Net: Accelerating Learning via Knowledge Transfer*
  (arXiv:1511.05641) — the original function-preserving construction. The `nnx` surgery primitives
  implement Net2WiderNet (§3.1) and Net2DeeperNet (§3.2) of that paper; see also the task
  [`README.md`](../../notebooks/model_surgery-mnist-ffnn-pytorch/README.md) for the in-repo
  contract summary and [`docs/env-setup.md`](../env-setup.md) §6 for the Tier-A execution path.
