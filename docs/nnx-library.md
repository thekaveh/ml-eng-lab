# 7. The NNx library

Almost every notebook in this repo trains, evaluates, or visualizes a model through
the same PyTorch toolkit: **NNx**. This page is the canonical in-repo reference for
what NNx is, how ml-eng-lab consumes it, how it was migrated from a git submodule to
a PyPI distribution, how to extend it, and where its known issues live. It is the
companion to the per-task deep-dives (e.g. `docs/notebooks/tabular_classification-iris-mlp-pytorch.md`),
which exercise the API surface end-to-end; read this page first when you need the
shape of the contract rather than the worked example.

## 7.1. What NNx is

NNx is a small PyTorch toolkit that factor out the boilerplate every notebook in
this lab would otherwise reinvent: dataset wrapping, model construction, training
loops, evaluation, serialization, and visualization. It is maintained in a sibling
repository and consumed here as a dependency — never edited in place (see §7.5).

Three names refer to the same project, and the distinction matters when you read
install instructions, import statements, or release notes:

| Name | Surface | Role |
|---|---|---|
| **NNx** (capitalized) | The GitHub project [`thekaveh/NNx`](https://github.com/thekaveh/NNx) | The upstream source repo; the canonical name used in prose and in this README |
| **`nnx`** (lowercase) | The importable Python package | What notebooks import: `from nnx import NNModel, NNParams, ...` |
| **`thekaveh-nnx`** | The PyPI distribution | What `pip install` resolves; the dependency name pinned in `requirements.txt` |

What the library provides, organized by the surfaces a notebook touches:

- **Dataset wrappers** — `NNDataset`, `NNTabularDataset`, `NNGraphDataset`. Wrap a
  tensor/DataFrame/graph payload and expose `train_loader` / `val_loader` /
  `test_loader` plus `input_dim` / `output_dim` metadata. They accept a
  `batch_sizes: tuple[Optional[int], Optional[int], Optional[int]]` argument so the
  caller picks the per-split batch size without bypassing the wrapper.
- **Model + training contract** — `NNModel`, `NNParams`, `NNModelParams`,
  `NNTrainParams`, `NNOptimParams`. Configuration objects (dataclasses) that make
  the model + training contract read as configuration over magic strings; enums
  (`Nets.FEED_FWD`, `Nets.TRANSFORMER`, `Losses.CROSS_ENTROPY`, `Optims.ADAM`,
  `Devices.CPU`) name the choices.
- **Generative modeling** — `GenerativeNNModel`, `TransformerNN`, `NNTransformerParams`,
  plus a BPE training helper (`train_bpe`) and tokenizer params
  (`NNTokenizerParams`) used by the two language-modeling notebooks.
- **Training history + serialization** — `NNRun`. The object returned by
  `model.train(...)`; carries per-iteration evaluation data points (`run.idps[i].train_edp`,
  `run.idps[i].val_edp`) and is also the serialization surface (`NNRun.load("best")`
  restores the best checkpoint from disk in a fresh session).
- **Visualization** — `VisUtils`. `multi_line_plot`, `confusion_matrix`, and
  related helpers that produce the convergence curves and confusion matrices seen
  throughout the deep-dives.
- **`[lm]` extra** — the optional dependency group that pulls the BPE tokenizer
  (`tokenizers`) and datasets backbone (`datasets`) needed by the two
  language-modeling notebooks. Without it, importing `train_bpe` /
  `NNTokenizerParams` raises `ImportError`.
- **Quantization / model surgery** — `nnx.quantize_int8`, `qat_train_step_factory`,
  `nnx.deepen`, `nnx.widen`. Used by `quantization-mnist-ffnn-pytorch` and
  `model_surgery-mnist-ffnn-pytorch` respectively.

The flat re-export shape is what makes notebook code read top-to-bottom: a single
`from nnx import (...)` block names every symbol the notebook uses, and the enums
make the contract self-describing (see
`docs/notebooks/tabular_classification-iris-mlp-pytorch.md` §8.1.5 for the canonical
example).

## 7.2. Consumption

NNx is consumed as an ordinary pinned PyPI dependency. The pin lives in
`requirements.txt`:

```
thekaveh-nnx[lm]==0.2.0
```

Three consequences worth keeping in mind:

1. **The `[lm]` extra is load-bearing for two notebooks.**
   `notebooks/text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb`
   and `notebooks/preference_alignment-toy-dpo-pytorch/notebook.ipynb` call
   `train_bpe` and reference `NNTokenizerParams`; without the extra, both fail at
   import with `ImportError` (issue #12). Every quick-start path in `README.md` §3
   installs `requirements.txt` verbatim, so the extra ships by default — but a
   caller who installs `thekaveh-nnx` without `[lm]` will hit this.

2. **Notebook imports are unchanged from the submodule era.** A notebook writes
   `from nnx import NNModel, NNParams, NNRun, ...` or `from nnx import
   NNTokenizerParams, train_bpe` regardless of distribution mechanism. The
   `nnx` package name is identical between the old git-submodule editable install
   and the new PyPI wheel — only the distribution name (`thekaveh-nnx`) differs.

3. **The pin is exact (`==0.2.0`), not a range.** This is deliberate: nnx is the
   load-bearing API surface for ~23 active notebooks, and a floating pin would
   let an upstream release silently shift a constructor signature under CI's fast
   lane (see `reference-nnx-dev-vs-pypi-drift` in the maintainer memory for the
   post-hoc case that motivated `test_nnx_constructor_calls_use_known_kwargs`).
   Bumping the pin is a deliberate act that triggers Tier-A papermill re-execution
   (see §7.4).

## 7.3. The 2026-06-14 PyPI migration

Before 2026-06-14, ml-eng-lab consumed NNx as a git submodule at `./nnx` with an
editable install (`-e ./nnx[lm]`). PR #19 swapped that for the PyPI wheel. The
migration is documented in `docs/FINDINGS-NNX.md`'s header note; the headline
shape:

| | Before (submodule) | After (PyPI) |
|---|---|---|
| Distribution name | `nnx` (editable, in-tree) | `thekaveh-nnx` (PyPI wheel) |
| Install mechanism | `-e ./nnx[lm]` in `requirements.txt`; `setup-in-jupyter.sh` bootstrapped the submodule | `thekaveh-nnx[lm]==0.2.0` in `requirements.txt`; no bootstrap script |
| Import paths | `from nnx import ...` | `from nnx import ...` (unchanged) |
| In-repo source tree | `./nnx/` tracked as a submodule | removed |
| Bump workflow | submodule-pointer bump PR | version-pin bump PR |
| Local-edit iteration | edit `./nnx/`, reinstall in place | clone `thekaveh/NNx` outside the tree, `pip install -e <path-to-clone>[lm]` |

The migration also closed issue #12 (`tokenizers` dep): under the submodule era,
the `[lm]` extra's `tokenizers` / `datasets` deps were satisfied transitively but
not declared; the PyPI wheel declares them properly, so a clean
`pip install thekaveh-nnx[lm]` resolves them without a second install step.

The delayed Tier-B/C caller fallout from the migration was swept up by PR #24
(2026-06-19): the Tier-A papermill gate missed `Utils`→`VisUtils` renames and an
`NNOptimParams` momentum signature change because no Tier-A notebook exercised
those surfaces — smoke-tier-b/c caught them. The lesson (recorded in the
maintainer memory as `reference-nnx-bump-caller-fixups`) is that a wide upstream
window requires the same post-bump audit as a submodule-pointer bump, just
substituting release notes for diffed pointers.

## 7.4. Extending NNx

NNx is not extended inside this repo. The workflow is always:

1. **Open a PR against [`thekaveh/NNx`](https://github.com/thekaveh/NNx)** with the
   new feature and a smoke test in the upstream test suite. For editable iteration
   during the design phase — before the feature is ready to land upstream — clone
   `thekaveh/NNx` *outside* the ml-eng-lab tree and `pip install -e <path-to-clone>[lm]`
   into your venv. The in-repo `import nnx` resolves to whichever copy
   `pip` installed last; cloning outside the tree avoids confusing the verifier's
   repo-state checks.

2. **Wait for the next NNx release cut.** A merged upstream PR is not consumable
   from ml-eng-lab until a new `thekaveh-nnx` wheel lands on PyPI. This is the
   single cost of the PyPI migration versus the submodule workflow — a submodule-
   pointer bump could consume any upstream commit; a version-pin bump can only
   consume a released wheel.

3. **Bump the pin in `requirements.txt`** (e.g.
   `thekaveh-nnx[lm]==0.2.0` → `thekaveh-nnx[lm]==0.2.1`) and open a PR here. CI
   re-runs automatically:
   - **Tier-A papermill** (`make run-tier-a` in CI) re-executes every Tier-A
     notebook against the new version. This is the primary regression gate for
     API-surface changes (constructor signatures, enum values, default
     arguments). PRs #15 through #20 established that a clean Tier-A run takes
     ~50-55 minutes on the Linux GH runner.
   - **`pytest-nnx-surface`** runs the static guards that catch constructor-call
     drift between what notebooks pass and what the released wheel accepts
     (`test_nnx_constructor_calls_use_known_kwargs`). This guard exists precisely
     because the Tier-A gate misses surfaces that no Tier-A notebook exercises —
     it was added after PR #26 surfaced post-hoc `NNGraphDataset(seed=)` drift.
   - **`make smoke-tier-b` and `make smoke-tier-c`** must be run manually when the
     NNx change touches graph or quantization surfaces (the Tier-A gate does not
     cover them — `torch_sparse` is Linux-only and Reddit-scale data does not fit
     on macOS). The same validation discipline applies as under the prior
     submodule-pointer-bump workflow; only the trigger (a version-pin diff vs. a
     submodule-pointer diff) has changed.
   - **Manual quantization validation** when the change touches the
     `quantize_int8` / `qat_train_step_factory` surface, since the quantization
     notebook is not in the Tier-A list.

The two non-negotiables, both learned the hard way:

- **Verify against the released wheel, not the dev checkout.** A local `import nnx`
  resolves to the `thekaveh/NNx` dev source, which is *ahead* of the released
  `thekaveh-nnx==0.2.0` wheel. A clean local re-execution therefore does not
  prove CI-compatibility. The dev-vs-PyPI drift case is real and recurs
  (`reference-nnx-dev-vs-pypi-drift`); `pip download thekaveh-nnx==X.Y.Z` into a
  scratch venv, install, and re-run there before pushing the bump.
- **Expect caller-side fixups when the upstream window is wide.** Local pytest +
  import smoke + `verify_repo.py` are insufficient when more than a single PR
  lands upstream between bumps; Tier-A papermill re-execution surfaces
  contract-fix breakage that the fast lane misses.

## 7.5. Findings & upstream

NNx is not edited from this repo. Fixes land upstream in
[`thekaveh/NNx`](https://github.com/thekaveh/NNx); ml-eng-lab records the known
issues and consumes the next release that addresses them.

The issue sink is `docs/FINDINGS-NNX.md`. Every API surprise, surprising default,
or documented-but-suboptimal behavior surfaced by the verify_repo loop or by a
notebook's workaround gets a numbered entry there, with: the surfacing notebook(s),
the exact behavior, the workaround in use, and the suggested upstream fix. The
file's header note also pins the 2026-06-14 migration context, so source paths it
cites (e.g. `nnx/src/nnx/nn/dataset/nn_dataset.py:24`) refer to the upstream repo,
not a local submodule.

The current findings, summarized — see `docs/FINDINGS-NNX.md` for the full text
and the suggested upstream fixes:

- **§1.1 — `NNDataset` default `batch_size` packs the whole train set into one
  batch.** Surprising for diffusion / MoE / transformer / JEPA tasks; the
  upstream `batch_sizes=` constructor arg (partial fix landed) gives a cleaner
  workaround than bypassing the wrapper with `DataLoader(...dataset, batch_size=N)`.
- **§1.2 — `nnx.deepen` is function-preserving only for `Activations.RELU`.**
  Identity-init insertion only preserves the forward for ReLU; the construction-
  time `ValueError` is clear, the constraint just isn't a one-liner to discover.
- **§1.3 — `NNTabularDataset` coerces targets to `torch.long` (classification-
  only).** Documented in the docstring; regression notebooks build the DataLoaders
  manually.
- **§1.4 — `EarlyStopping(monitor=...)` default is `"val_edp.error"`, which does
  not exist for regression EDPs.** Regression callers must pass
  `monitor="val_edp.loss"` explicitly.
- **§1.5 — `NNRun.save()` prints an absolute path, leaking the execution
  environment layout.** Active notebook outputs can carry machine-local paths;
  the `E13.stale_active_notebook_path` verifier rule keeps them clean in-repo, and
  the upstream fix is to print a `cwd`-relative run path.

When you trip over a new NNx behavior during notebook authoring or a re-execution
pass, the right move is to add a finding to `docs/FINDINGS-NNX.md` (with the
surfacing notebook, the exact reproduction, and the suggested upstream fix) and
keep the in-repo notebook working with a local workaround — not to fork or patch
nnx in this tree. The next NNx release cut that addresses the finding is then a
routine version-pin bump under §7.4.
