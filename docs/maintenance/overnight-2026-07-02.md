# Overnight Maintenance Log - 2026-07-02

Branch: `codex/overnight-maintenance`
Upstream: `origin/codex/overnight-maintenance`
Spec: `/Users/kaveh/.agents/skills/overnight-maintenance/maintenance-spec.md`
Parameters: `PASSES=25`, `MAX_PASSES=75`, `PUSH=yes`, `NUMBERED_DOCS=yes`

## Coverage Decisions

Applied checks:
- Baseline repository scan: Python code, notebooks, tests, docs, scripts, CI, Docker, devcontainer, dependency manifests.
- Documentation indexing and hierarchical numbering because `NUMBERED_DOCS=yes`.
- End-to-end execution-flow tracing for notebooks, scripts, Makefile targets, and test/tooling entry points.
- Design-pattern and refactoring opportunity scan.
- Error-handling, configuration, secrets, versioning, build, supply-chain, complexity, dependency-contract, and examples/library-fit checks.

Skipped checks:
- HTTP/RPC handler contracts: N/A, no service handlers found in this repository.
- Database and migration hygiene: N/A, no persistent database or migration files found.
- i18n/a11y UI checks: N/A, no user-facing frontend application found.
- Multi-language parity for active surfaces: N/A for active notebooks and Python helpers; archived CodexGLUE material is reviewed as archive hygiene, not an active supported surface.
- Performance budget benchmarking: N/A unless a repository-declared performance budget is found during pass review.

## Pass History

| Pass | Type | Issues | Coverage Evidence | Result |
| --- | --- | ---: | --- | --- |
| 1 | genuine | 18 | Inventory gathered: git state, tracked files, 51 notebooks / 61 Python files / 30 Markdown files; baseline and post-fix `ruff check .`; `pytest tests/`; `pytest tests/nnx_surface`; `python scripts/verify_repo.py --check all --fast`; `pip-audit -r requirements.txt -r torch-requirements.txt`; fresh-eyes report-only subagents for docs, notebooks, tooling/security/dependencies, and Python/tests/complexity. | Non-zero pass; valid findings fixed or explicitly deferred |

## Validation Log

- `ruff check .` passed before and after the pass-1 fix batch.
- Baseline `pytest tests/ -v` passed with `291 passed, 3 skipped`; post-fix full suite passed with `325 passed, 3 skipped, 19 warnings`.
- Baseline `pytest tests/nnx_surface -v` passed with `258 passed, 3 skipped`; post-fix NNx-surface coverage is included in the full `pytest tests/ -v` run.
- `python scripts/verify_repo.py --check all --fast` passed before and after the fix batch with `0 errors, 5 warnings` (`torch_sparse` unavailable in the local verifier environment for four Reddit phase3 notebooks, plus missing optional `shellcheck`).
- `bash -n scripts/start-jupyterhub.sh` passed.
- YAML parsing passed for `.github/workflows/ci.yml` and `deploy/genai-vanilla-jupyterhub.override.yml`.
- `scripts/start-jupyterhub.sh --help` exited with status 1 by design in this checkout and printed the intended uninitialized-submodule recovery command.
- `git diff --check` passed.
- `pip-audit -r requirements.txt -r torch-requirements.txt` still reports 23 known vulnerabilities across `torch==2.4.1`, `pytorch-lightning==2.4.0`, and `nltk==3.9.4`; this is recorded in `docs/dependency-contracts.md` and remains a coordinated dependency-upgrade follow-up rather than a local green check.

## Issue Log

| ID | Severity | Category | Location | Description | Status | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| OM-001 | High | §3.16 security / §3.14 CI | `.github/workflows/ci.yml` | CI ran PR-controlled code without explicit least-privilege permissions and with persisted checkout credentials. | Fixed | Added top-level `permissions: contents: read`; set `persist-credentials: false` on checkout steps; workflow YAML parsed successfully. |
| OM-002 | High | §3.16 security / §3.26 configuration | `scripts/start-jupyterhub.sh`, `deploy/genai-vanilla-jupyterhub.override.yml`, runtime docs | Wrapper mounted host `~/.ssh` into JupyterHub by default. | Fixed | Host SSH mount now opt-in via `HOST_SSH_DIR`; docs updated; shell syntax and override YAML validated. |
| OM-003 | High | §3.16 dependency security / §3.31 consumed contracts | `requirements.txt`, `torch-requirements.txt`, `docs/dependency-contracts.md` | `pip-audit` reported 23 known vulnerabilities in pinned `torch`, `pytorch-lightning`, and `nltk`; exception state was undocumented. | Fixed/documented | Added dependency contract ledger with audit snapshot and upgrade criteria; full upgrade deferred to coordinated Torch stack pass. |
| OM-004 | Medium | §3.29 reproducibility | Docker/devcontainer/PyPI/NLP assets | Build inputs are tag-pinned/ranged and external NLP assets are not checksum locked. | Documented/deferred | Ledger records current contract and stricter reproducibility upgrade path; full lockfile/digest migration deferred due broad dependency blast radius. |
| OM-005 | Medium | §3.14 CI/tooling | `.github/workflows/ci.yml` | `make verify` was local-only, despite enforcing repo invariants. | Fixed | Added `verify-repo` CI job running `make verify`; workflow YAML parsed successfully. |
| OM-006 | Low | §3.17 bootstrap tracing | `scripts/start-jupyterhub.sh` | Wrapper only checked that `vendor/genai-vanilla` directory existed, not whether the submodule files were initialized. | Fixed | Wrapper now checks for `start.sh` and `docker-compose.yml`; local uninitialized-submodule run prints `git submodule update --init --recursive` and exits 1 by design. |
| OM-007 | Low | §3.6 docs | `README.md`, `.devcontainer/devcontainer.json` | Codespaces wording overpromised all active notebooks despite manual-only quantization. | Fixed | Wording now distinguishes 21 task folders, 29 active notebooks, 28 tier-covered runnable notebooks, and the manual-only quantization notebook. |
| OM-008 | Medium | §3.6 changelog hygiene | `CHANGELOG.md` | `[Unreleased]` had duplicate `### Fixed` sections. | Fixed | Merged duplicate Fixed block. |
| OM-009 | Low | §3.9 numbered docs | `archive/README.md` | Archive README headings were unnumbered under `NUMBERED_DOCS=yes`. | Fixed | Renumbered archive headings hierarchically. |
| OM-010 | Low | §3.8 link hygiene | `CHANGELOG.md` | Historical broken-link example was itself a live broken Markdown link. | Fixed | Rendered as literal code text. |
| OM-011 | Medium | §3.4 correctness / §3.13 tests | `image_classification-mnist-ffnn-numpy/utils.py` | `one_hot_encode` assumed dense zero-based integer labels and ignored explicit class order. | Fixed | Added `tests/test_numpy_utils.py`; focused test and full `pytest tests/ -v` passed. |
| OM-012 | Medium | §3.15 hygiene / §3.17 generated-helper trace | `scripts/rewrite_imports.py` | Simple import migration rewrote comments and string literals. | Fixed | Added regression test; focused test and full `pytest tests/ -v` passed. |
| OM-013 | Medium | §3.17 notebook trace / §3.31 PyG contract | `node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb` | `ToSparseTensor()` dropped `edge_index` by default while later cells rely on `edge_index`. | Fixed | Added NNx surface guard; focused real-notebook test and full `pytest tests/ -v` passed. |
| OM-014 | Medium | §3.17 notebook reproducibility | Reddit phase2/phase3 notebooks | New upstream NNx supports `NNGraphDataset(seed=...)`, but pinned `thekaveh-nnx==0.2.0` does not. | Deferred | Do not add `seed=` until requirements bump; track with next NNx upgrade. |
| OM-015 | Low | §3.32 examples/library-fit | Reddit notebooks | Some Reddit notebooks still use deep NNx imports instead of top-level public facade; phase3 has stale manual-PyG imports. | Deferred | Needs notebook source sweep and Tier-C baseline/E5 handling; not mixed into security/doc batch. |
| OM-016 | Medium | §3.13 tests / §3.15 hygiene | `tests/test_verify_repo.py` | Some verifier tests write temporary files inside the real repo tree. | Deferred | Requires verifier root/config injection design; existing tests pass and cleanup with `try/finally`. |
| OM-017 | Medium | §3.17 CLI trace | `scripts/verify_repo.py` | Config/YAML loads at import time before argparse `--help`. | Deferred | Requires targeted CLI/import refactor; current `--help` test passes in provisioned environment. |
| OM-018 | Low | §3.13 tests | `tests/nnx_surface/test_notebook_api_surface.py` | `_active_notebooks()` scans all files, so untracked scratch notebooks can affect local tests. | Deferred | Low local-only risk; candidate for next pass. |
