# Contributing

A short guide for adding new task folders and modifying shared code in this lab. For the assistant-facing conventions, see [`CLAUDE.md`](CLAUDE.md).

## Workflow

1. Read [`CLAUDE.md`](CLAUDE.md) — it captures the conventions tersely.
2. Open a feature branch off `main`.
3. Make your change.
4. Run `python scripts/verify_repo.py --check all --fast` — must exit 0 (no error-severity findings; warnings are OK).
5. If you touched a notebook, re-run it (Tier-A: `make run-tier-a`; Tier-B: `make smoke-tier-b`; Tier-C: `make smoke-tier-c`). Tier-C source notebooks must remain byte-equal to the `pre-cleanup-baseline` tag — verify check E5 enforces this.
6. Open a PR. CI runs Tier-A automatically; Tier-B/C run on schedule and on `workflow_dispatch`.

## Adding a new task folder

Convention: top-level folder named `[task]-[dataset]-[model]-[framework]/`.

1. Survey [`nnx/src/nnx/`](nnx/src/nnx/) for reusable primitives.
2. Identify gaps. If you need new primitives, **land them in [`thekaveh/NNx`](https://github.com/thekaveh/NNx) first** (branch in `./nnx`, commit, push), then bump the submodule pointer here.
3. Scaffold the new task folder with a `README.md` (use [`node_classification-reddit-gnn-pyg/README.md`](node_classification-reddit-gnn-pyg/README.md) as template) and notebook(s).
4. Add the notebook(s) to `REQUIRED_SECTIONS` in [`scripts/verify_repo.py`](scripts/verify_repo.py) — or accept the default 6-section requirement.
5. If Tier-A, add the notebook path to the `tier_a` tuple in `check_execution` (same file) and to `TIER_A` in [`Makefile`](Makefile).
6. Update the root README's task table.
7. Tick the box on the root README roadmap.

## Modifying shared code

- **`nnx/` is a submodule.** Don't make code changes directly to the submodule's tree from this repo without an upstream commit. See "Workflow for adding a new task" in [`CLAUDE.md`](CLAUDE.md).
- **`vendor/genai-vanilla/` is vendored.** Don't edit. The ml-specific compose override lives in [`deploy/`](deploy/).
- **`archive/` is read-only.** Preserved Aug-2023 work.

## One concern per PR

- Don't bundle unrelated cleanup with a feature change.
- Tier-C notebook re-execution belongs in its own PR if you ever need to (rare; preserved outputs are intentional).
