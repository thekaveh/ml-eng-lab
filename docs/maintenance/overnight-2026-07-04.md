# Overnight Maintenance Log - 2026-07-04

Branch: `codex/overnight-maintenance`
Upstream: `origin/codex/overnight-maintenance`
Spec: external `overnight-maintenance/maintenance-spec.md` skill spec
Parameters: `PASSES=20`, `MAX_PASSES=50`, `PUSH=yes`, `NUMBERED_DOCS=yes`

## 1. Coverage Decisions

Applied checks:
- Baseline repository scan: Python code, notebooks, tests, docs, scripts, CI, Docker, devcontainer, dependency manifests.
- Documentation indexing and hierarchical numbering because `NUMBERED_DOCS=yes`.
- End-to-end execution-flow tracing for notebooks, scripts, Makefile targets, generated docs, and verifier entry points.
- Documentation-surface review for README, MkDocs, repository metadata, and the project wiki/Pages state.
- Error-handling, configuration, secrets, versioning, build, supply-chain, complexity, dependency-contract, and examples/library-fit checks.

Skipped checks:
- HTTP/RPC handler contracts: N/A, no service handlers found in this repository.
- Database and migration hygiene: N/A, no persistent database or migration files found.
- i18n/a11y UI checks: N/A, no user-facing frontend application found.
- Multi-language parity for active surfaces: N/A for active notebooks and Python helpers; archived CodexGLUE material is reviewed as archive hygiene, not an active supported surface.
- Performance budget benchmarking: N/A unless a repository-declared performance budget is found during pass review.

## 2. Pass History

| Pass | Type | Issues | Coverage Evidence | Result |
| --- | --- | ---: | --- | --- |
| 1 | genuine | 12 | Fresh-eyes report-only subagents reviewed active notebooks/examples, published docs surfaces, tooling/security/reproducibility, and complexity/tests; local fixes covered orphaned notebook-reorganization records, nested-doc verifier coverage, subprocess timeout defaults, PyG runtime canary accuracy, NNx editable-install contract boundaries, and genai-vanilla submodule contract ledgering. Validation: focused red/green verifier tests; `python scripts/verify_repo.py --check all --fast` (`0 errors, 4 warnings`); `make docs-build`; `ruff check . --no-cache`; `pytest tests/ -v` (`381 passed, 3 skipped, 17 warnings`); `make check-tier-a-clean`; `shellcheck scripts/start-jupyterhub.sh`; `python -m py_compile scripts/verify_repo.py`; `git diff --check`. | Non-zero pass; fixes applied; zero-issue streak reset; remaining notebook/API-fit, output-hash, Pages, CI-signal, and dependency-refresh items deferred with rationale |

## 3. Validation Log

- `pytest tests/test_verify_repo.py::test_structure_s3_checks_nested_docs_markdown_links -q` failed before the docs-iterator fix and passed after it.
- `pytest tests/test_verify_repo.py::test_run_helper_supplies_default_timeout -q` failed before adding the default timeout and passed after it.
- `pytest tests/test_verify_repo.py::test_runtime_available_requires_pyg_extension_stack -q` failed before expanding the PyG canary and passed after it.
- `python scripts/verify_repo.py --check docs --fast`: 0 findings.
- `python scripts/verify_repo.py --check all --fast`: 0 errors, 4 warnings for runtime-container-only `torch_sparse` imports in Reddit phase3 notebooks.
- `make docs-build`: passed with the upstream Material for MkDocs 2.0 warning.
- `ruff check . --no-cache`: passed.
- `pytest tests/ -v`: 381 passed, 3 skipped, 17 warnings.
- `make check-tier-a-clean`: passed.
- `shellcheck scripts/start-jupyterhub.sh`: passed.
- `python -m py_compile scripts/verify_repo.py`: passed.
- `git diff --check`: passed.

## 4. Issue Log

| ID | Severity | Category | Location | Description | Status | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| OM-001 | Medium | §3.6 docs / §3.8 documentation indexing | `docs/superpowers/*`, `docs/maintenance/*`, `README.md`, `mkdocs.yml` | The notebook-reorganization design and implementation records were tracked under ignored `docs/superpowers/` paths and were not discoverable from README or the generated docs navigation. | Fixed | Moved the records to `docs/maintenance/`, indexed them in README and MkDocs nav, and validated docs/verifier/build paths. |
| OM-002 | Low | §3.9 numbered docs | `docs/maintenance/notebooks-reorganization-*.md` | The surfaced maintenance records used H2 headings such as `1.1.` even though the current D9 verifier expects H2 headings to use one numeric component. | Fixed | Normalized their H1/H2 structure; `python scripts/verify_repo.py --check docs --fast` now reports 0 findings. |
| OM-003 | Medium | §3.8 link hygiene / §3.13 tests | `scripts/verify_repo.py`, `tests/test_verify_repo.py` | Markdown link and terminology sweeps only covered shallow `docs/*.md`, so nested maintenance docs could drift outside S3/D8 coverage. | Fixed | Added a red/green nested-doc broken-link regression and changed the in-scope docs iterator to include nested non-superpowers docs. |
| OM-004 | Medium | §3.23 error handling / §3.17 CLI trace | `scripts/verify_repo.py`, `tests/test_verify_repo.py` | Short external verifier subprocesses could run without bounded timeouts when callers omitted an explicit timeout. | Fixed | Added a default subprocess timeout, applied it to `git ls-files`, preserved long notebook-target overrides, and added regression coverage. |
| OM-005 | Medium | §3.17 notebook trace / §3.31 PyG contract | `scripts/verify_repo.py`, `tests/test_verify_repo.py` | The full-execution runtime canary checked only `torch` and `torch_geometric`, missing PyG extension modules required by graph notebooks. | Fixed | Expanded the canary to `torch_sparse`, `torch_scatter`, `torch_cluster`, `torch_spline_conv`, and `pyg_lib`; added a false-positive regression. |
| OM-006 | High | §3.31 consumed contracts / §3.13 tests | `tests/nnx_surface/test_notebook_api_surface.py`, `docs/dependency-contracts.md` | Local NNx surface tests can validate an editable sibling checkout instead of exact pinned `thekaveh-nnx[lm]==0.2.0` without that nuance being recorded. | Documented/deferred | Ledger now records the PyPI pin as canonical, the editable override boundary, and the command to detect editable installs; enforcing failure for editable local dev is deferred to an explicit environment-policy change. |
| OM-007 | Medium | §3.31 consumed contracts / §3.17 runtime trace | `.gitmodules`, `vendor/genai-vanilla`, `scripts/start-jupyterhub.sh`, `deploy/genai-vanilla-jupyterhub.override.yml`, `docs/dependency-contracts.md` | The vendored genai-vanilla submodule contract and behind-upstream state were implicit instead of ledgered. | Fixed/documented | Added exact submodule pin, upstream SHA observed on 2026-07-04, wrapper/override expectations, and upgrade criteria to the dependency contract ledger. |
| OM-008 | Medium | §3.32 examples/library-fit | Active notebooks using `nnx.nn.*`, `nnx.utils`, and `nnx.vis_utils` imports | Several active examples still teach deep NNx module paths even when the top-level `nnx` facade exports the symbols. | Deferred | Migration requires a notebook source/output sweep and Tier-C baseline handling, so it remains a separate notebook modernization pass. |
| OM-009 | Medium | §3.32 examples/library-fit | `notebooks/diffusion-mnist-ddpm-pytorch/`, `notebooks/moe-fmnist-mixture-of-experts-pytorch/`, `notebooks/self_supervised-fmnist-jepa-pytorch/` | Three notebooks still rebuild loaders manually from `NNDataset` internals instead of using `NNDataset(batch_sizes=...)`. | Deferred | Needs notebook execution/output refresh to avoid stale committed outputs; keep grouped with the NNx facade-import modernization. |
| OM-010 | Medium | §3.17 notebook trace / §3.13 tests | Active notebook outputs, `scripts/verify_repo.py` | Stale-output detection is pre-positioned on optional `metadata.source_hash`, but current output-bearing cells have no hash marker, making the check a no-op today. | Deferred | Requires a notebook execution hook and one refresh cycle to stamp hashes before missing hashes can become warnings or errors. |
| OM-011 | High | §3.33 published docs surfaces | GitHub Pages, repository About homepage, project wiki | The generated docs site is scaffolded but the live Pages URL still returns 404 until the workflow lands on the default branch; About therefore points to README rather than the dead Pages URL. | Deferred | Cannot complete from the maintenance branch without modifying default-branch live deployment state; keep README homepage as safer temporary metadata until Pages is live, then update About/wiki links. |
| OM-012 | Low | §3.14 CI/tooling | `.github/workflows/ci.yml`, GitHub Actions branch runs | Local validation is current, but GitHub Actions has no fresh branch CI run for this head. | Deferred | Trigger/open PR workflow after pushing this branch so GitHub validates the pushed commit. |
