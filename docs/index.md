# 1. Overview

`ml-eng-lab` is a portfolio of self-contained machine-learning notebook experiments. Active
experiments live under `notebooks/`; archived CodeXGLUE experiments live under
`notebooks/archive/`; repository tooling keeps notebook execution, documentation, CI, and
dependency contracts aligned.

This page is the entry point for the generated documentation site. The site is one of three
documentation surfaces — repository, site, and wiki — all derived from a single canonical source
tree under `docs/` by the pipeline in `scripts/docs/`. Section 2 below describes the surfaces;
[the System & context view](architecture.md) draws the full picture.

## 1. Repository map

- `notebooks/` contains twenty-one active task directories and twenty-nine active notebooks.
- `notebooks/archive/` contains preserved Aug-2023 CodeXGLUE summarization experiments.
- `scripts/verify_repo.py` is the fast structural, documentation, and notebook-surface verifier.
- `scripts/docs/` owns the three-surface documentation pipeline (manifest, transforms,
  renderers, checker).
- `Makefile` owns notebook execution tiers and local validation targets.
- `docs/` holds the canonical documentation sources plus maintenance logs and findings.
- `.github/workflows/` contains CI and documentation publishing workflows.

The root `README.md` is the day-to-day entry point for contributors — it carries the task index,
quick-start paths, and the standard make targets. This generated site is the focused reference
surface that sits behind the README.

## 2. Documentation surfaces

The lab maintains three synchronized documentation surfaces, all derived from one canonical
source tree so the three never drift:

| Surface | Source | Rendered by | Audience |
|---|---|---|---|
| **Repository** | `docs/*.md` (checked in) | GitHub markdown rendering | Contributors browsing the repo |
| **Site** | `generated/site/` | MkDocs Material (`mkdocs build`) | Public readers of the published site |
| **Wiki** | `generated/wiki/` | GitHub wiki rendering | Readers who prefer the wiki navigation |

The manifest at `docs/manifest.yaml` is the single source of truth for the hierarchy, numbering,
and page set. `scripts/docs/build_docs.py` consumes the manifest and emits both generated
surfaces; `scripts/docs/check_docs.py` gates CI on self-containment, completeness, placeholders,
and determinism. The canonical sources are written once; every surface is a transform of them.

## 3. Recommended reading path

- [System & context view](architecture.md) for the repository context, the system diagram,
  and the three-surface pipeline.
- [Tabular classification — Iris MLP](notebooks/tabular_classification-iris-mlp-pytorch.md)
  for the exemplar comprehensive deep-dive — the canonical walk-through of one notebook end to
  end (problem, math, architecture, code, results, pitfalls, extensions).

Additional notebook deep-dives will be added under `docs/notebooks/` as the manifest grows; the
catalog above is the current page set.
