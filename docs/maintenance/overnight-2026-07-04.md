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
| 2 | genuine | 11 | Fresh report-only subagents reviewed notebooks/examples, docs/published surfaces, tooling/security/reproducibility, and complexity/tests against the current tree; local trace covered `.github/workflows/*`, `Makefile` papermill targets, `docs/index.md`, `docs/dependency-contracts.md`, `docs/FINDINGS-NNX.md`, `docs/diagrams/*`, NumPy notebook helper modules, verifier constants, and live repository metadata. Validation: red/green `pytest tests/test_numpy_layers.py -v`; `pytest tests/test_numpy_layers.py tests/test_verify_repo.py -q` (`60 passed`); workflow YAML parse; `python scripts/verify_repo.py --check docs --fast` (`0 findings`); `ruff check . --no-cache`; `python -m py_compile scripts/verify_repo.py`; `python scripts/verify_repo.py --check all --fast` (`0 errors, 4 known warnings`); `make docs-build`; `pytest tests/ -v` (`384 passed, 3 skipped, 17 warnings`); `make check-tier-a-clean`; `shellcheck scripts/start-jupyterhub.sh`; `pip-audit -r requirements.txt -r torch-requirements.txt` (`23 known vulnerabilities`, ledgered); `git diff --check`. | Non-zero pass; fixes applied for docs surfacing, tooling timeouts, action pins, NumPy layer validation, and stale docs wording; zero-issue streak reset; live Pages, notebook modernization, output hashing, dependency lockfiles, and complexity refactors remain deferred |
| 3 | genuine | 3 | Fresh report-only subagents reviewed active notebooks/examples, docs/diagrams/published surfaces, tooling/security/dependency contracts, and Python code/tests/complexity; local trace covered active notebook papermill metadata, README library-fit wording, genai-vanilla pinned and upstream contracts, GitHub Actions pins, live Pages/About metadata, tracked data/runs/cache artifacts, stale `ml-lab` references, and standard validation. Validation: red/green `pytest tests/test_verify_repo.py::test_e14_flags_tmp_papermill_output_path -q`; `python scripts/verify_repo.py --check execution --fast` (`0 findings` after cleanup); `python scripts/verify_repo.py --check all --fast` (`0 errors, 4 known warnings`); `ruff check . --no-cache`; `make docs-build`; `pytest tests/ -v` (`385 passed, 3 skipped, 17 warnings`); `python -m py_compile scripts/verify_repo.py`; `shellcheck scripts/start-jupyterhub.sh`; `bash -n scripts/start-jupyterhub.sh`; `pip-audit -r requirements.txt -r torch-requirements.txt` (`23 known vulnerabilities`, ledgered); notebook JSON parse; `git diff --check`. | Non-zero pass; fixes applied for stale papermill metadata, stale README wording, and genai-vanilla upstream ledger drift; zero-issue streak reset |
| 4 | genuine | 4 | Fresh report-only subagents reviewed active notebooks/examples/archive placement, documentation/published surfaces, tooling/security/dependency contracts, and Python code/tests/complexity; local trace covered active/archive notebook inventory, relative `./data` and `./runs` references, papermill metadata, tracked data/runs/cache artifacts, stale `ml-lab` references, GitHub Actions pins, live Pages/About/wiki state, genai-vanilla submodule wording, Docker bootstrap ledger wording, Reddit phase-2 run selection, quantization facade coverage, and standard validation. Validation: red/green `pytest tests/nnx_surface/test_notebook_api_surface.py::test_phase2_notebook4_ranks_local_runs_not_cross_experiment_registry -q`; `pytest tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py -q`; notebook JSON parse for the Reddit phase-2 notebook. | Non-zero pass; fixes applied for submodule/ledger wording, Reddit local-run ranking, stale Reddit outputs, and quantization facade coverage; zero-issue streak reset; live Pages/default-branch docs, `NNDataset(batch_sizes=...)` modernization, output hashing, CI signal, bootstrap lockfile pinning, and verifier complexity remain deferred |

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
- `python scripts/verify_repo.py --check all --fast`: 0 errors, 4 known warnings for runtime-container-only `torch_sparse` imports in Reddit phase3 notebooks.
- `make docs-build`: passed with the upstream Material for MkDocs 2.0 warning.
- `pytest tests/ -v`: 384 passed, 3 skipped, 17 warnings.
- `make check-tier-a-clean`: passed.
- `shellcheck scripts/start-jupyterhub.sh`: passed.
- `pip-audit -r requirements.txt -r torch-requirements.txt`: 23 known vulnerabilities in the same ledgered `torch`, `pytorch-lightning`, and `nltk` packages.
- `pytest tests/test_numpy_layers.py -v`: failed before validating supplied `LinearLayer` parameter shapes, then passed after the constructor fix (`3 passed`).
- `python - <<'PY' ... yaml.safe_load(...)`: parsed `.github/workflows/pages.yml` and `.github/workflows/ci.yml`.
- `pytest tests/test_numpy_layers.py tests/test_verify_repo.py -q`: 60 passed.
- `python scripts/verify_repo.py --check docs --fast`: 0 findings.
- `ruff check . --no-cache`: passed.
- `python -m py_compile scripts/verify_repo.py`: passed.
- `git diff --check`: passed.
- `pytest tests/test_verify_repo.py::test_e14_flags_tmp_papermill_output_path -q`: failed before adding the E14 verifier guard, then passed after it.
- `python scripts/verify_repo.py --check execution --fast`: first reported 17 `E14.tmp_papermill_output_path` warnings, then 0 findings after stripping top-level stale papermill metadata.
- `curl -I -L https://thekaveh.github.io/ml-eng-lab/`: still returned HTTP 404, matching deferred OM-011.
- `gh repo view thekaveh/ml-eng-lab --json name,description,homepageUrl,repositoryTopics,defaultBranchRef,url`: default branch `main`, description/topics current, homepage still README pending live Pages.
- `git ls-remote --tags` for pinned GitHub Actions: reviewed tag SHAs still match `.github/workflows/*` and `docs/dependency-contracts.md`.
- `git ls-remote https://github.com/thekaveh/genai-vanilla.git refs/heads/main`: upstream `main` is `b0bce0fc4e9d2bb282fbc6c97631f3e37233e24e`.
- `python scripts/verify_repo.py --check all --fast`: 0 errors, 4 known warnings for runtime-container-only `torch_sparse` imports in Reddit phase3 notebooks.
- `ruff check . --no-cache`: passed.
- `make docs-build`: passed with the upstream Material for MkDocs 2.0 warning.
- `pytest tests/ -v`: 385 passed, 3 skipped, 17 warnings.
- `python -m py_compile scripts/verify_repo.py`: passed.
- `shellcheck scripts/start-jupyterhub.sh`: passed.
- `bash -n scripts/start-jupyterhub.sh`: passed.
- `pip-audit -r requirements.txt -r torch-requirements.txt`: 23 known vulnerabilities in the same ledgered `torch`, `pytorch-lightning`, and `nltk` packages.
- Notebook JSON parse across `notebooks/**/*.ipynb`: passed.
- `git diff --check`: passed.
- `pytest tests/nnx_surface/test_notebook_api_surface.py::test_phase2_notebook4_ranks_local_runs_not_cross_experiment_registry -q`: failed before replacing `NNRun.all()` in Reddit phase2 notebook 4, then passed after ranking the notebook-local `runs` list and clearing stale dependent outputs.
- `pytest tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py -q`: 3 passed, covering quantization facade signatures plus backend-gated PTQ/QAT smoke.
- `python -m json.tool notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb >/dev/null`: passed.
- `ruff check . --no-cache`: first flagged the now-unused `NNRun` import in Reddit phase2 notebook 4, then passed after removing it.
- `python scripts/verify_repo.py --check all --fast`: 0 errors, 4 known warnings for runtime-container-only `torch_sparse` imports in Reddit phase3 notebooks.
- `pytest tests/ -v`: 389 passed, 3 skipped, 17 warnings.
- `make docs-build`: passed with the upstream Material for MkDocs 2.0 warning.
- `shellcheck scripts/start-jupyterhub.sh && bash -n scripts/start-jupyterhub.sh`: passed.
- `git diff --check`: passed.
- `python -m py_compile scripts/verify_repo.py`: passed.
- `pip-audit -r requirements.txt -r torch-requirements.txt`: 23 known vulnerabilities in the same ledgered `torch`, `pytorch-lightning`, and `nltk` packages.
- Notebook JSON parse across `notebooks/**/*.ipynb`: parsed 51 notebooks.
- Tracked `data/`/`runs/` scan and generated-cache scan: no tracked matches.
- `make check-tier-a-clean`: passed after staging the intentional Reddit phase2 notebook artifact update.

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
| OM-011 | High | §3.33 published docs surfaces | GitHub Pages, default-branch README, repository About homepage, project wiki | The generated docs site is scaffolded and branch-local docs are current, but live Pages still returns 404 and the default-branch README/wiki/About surfaces remain behind this branch until the workflow and docs land on `main`. | Deferred | Cannot complete from the maintenance branch without modifying default-branch live deployment state; keep README homepage as safer temporary metadata until Pages is live, then update About/wiki links. |
| OM-012 | Low | §3.14 CI/tooling | `.github/workflows/ci.yml`, GitHub Actions branch runs | Local validation is current, but GitHub Actions has no fresh branch CI run for this head. | Deferred | Trigger/open PR workflow after pushing this branch so GitHub validates the pushed commit. |
| OM-013 | Medium | §3.8 docs indexing / §3.33 published surfaces | `docs/index.md` | The generated-docs homepage still routed "Maintenance Log" readers to the July 2 hard-cap log instead of the active July 4 maintenance record. | Fixed | Updated the recommended reading path to name the current July 4 log and retain the July 2 log as historical context; docs verifier reports 0 findings. |
| OM-014 | Medium | §3.6 generated artifacts / §3.33 diagrams | `docs/diagrams/`, `docs/architecture.md`, `README.md`, `mkdocs.yml` | Checked-in diagram HTML artifacts had no provenance or regeneration contract in the docs navigation. | Fixed | Added `docs/diagrams/README.md`, linked it from architecture docs and README, and added it to MkDocs navigation. |
| OM-015 | Low | §3.8 documentation tone | `CHANGELOG.md` | A decorative checkmark glyph remained in a changelog entry, contradicting the repository's plain professional docs tone. | Fixed | Replaced the glyph with prose; `git diff --check` stays clean. |
| OM-016 | Medium | §3.29 supply-chain reproducibility | `.github/workflows/ci.yml`, `.github/workflows/pages.yml`, `docs/dependency-contracts.md` | Several GitHub Actions were exact-SHA pinned to older reviewed majors while newer stable major tags existed. | Fixed | Resolved current tag SHAs with `git ls-remote --tags`, updated workflow pins/comments, and ledgered the reviewed action tags and upgrade process. |
| OM-017 | Medium | §3.14 CI/tooling / §3.23 resilience | `.github/workflows/pages.yml` | Pages build and deploy jobs lacked explicit `timeout-minutes`, so hung doc deploys could consume the platform default. | Fixed | Added 15-minute build and 10-minute deploy caps; workflow YAML parses successfully. |
| OM-018 | Medium | §3.17 notebook execution / §3.23 resilience | `Makefile`, `docs/dependency-contracts.md` | Direct papermill targets had no `--start-timeout` or `--execution-timeout` bounds. | Fixed | Added centralized `PAPERMILL_START_TIMEOUT` and `PAPERMILL_EXECUTION_TIMEOUT` variables and passed them to Tier-A/B/C notebook targets; documented the CLI contract. |
| OM-019 | Medium | §3.29 build reproducibility | `Makefile`, `Dockerfile`, `docs/dependency-contracts.md` | Bootstrap paths still upgrade/install pip and setuptools without exact pins. | Documented/deferred | Added the bootstrap tooling gap to the dependency ledger; full pinning remains grouped with the lockfile/base-image digest pass because it changes every environment creation path. |
| OM-020 | Low | §3.32 examples/library-fit docs | `docs/FINDINGS-NNX.md` | The NNx findings doc overstated the `NNDataset` loader-bypass set by counting TinyShakespeare, which intentionally uses a custom sequence-window dataset. | Fixed | Updated the finding to distinguish the three `NNDataset` bypasses from the TinyShakespeare custom language-modeling dataset. |
| OM-021 | High | §3.4 correctness / §3.13 tests | `notebooks/image_classification-mnist-ffnn-numpy/linear_layer.py`, `tests/test_numpy_layers.py` | `LinearLayer` accepted supplied `W`/`b` arrays with incompatible shapes, then failed later or silently violated constructor dimensions. | Fixed | Added failing shape-regression tests first, then explicit constructor `ValueError` checks; focused tests now pass. |
| OM-022 | Medium | §3.30 code complexity | `scripts/verify_repo.py`, `scripts/rewrite_imports.py` | Maintenance scripts still exceed ideal complexity thresholds after the current pass. | Deferred | Refactoring is material and behavior-sensitive; keep it as a separate red/green script-maintainability pass rather than mixing it with workflow/docs/notebook-helper fixes. |
| OM-023 | Low | §3.10 dead code | `notebooks/image_classification-mnist-ffnn-numpy/consts.py`, `scripts/verify_repo.py` | Unused constants remained after the notebook reorganization and prior verifier refactors. | Fixed | Removed unused NumPy dataset-stat constants and stale verifier directory constants; focused verifier tests and `py_compile` pass. |
| OM-024 | Medium | §3.17 notebook trace / §3.13 tests | Active notebook `metadata.papermill`, `scripts/verify_repo.py`, `tests/test_verify_repo.py` | Seventeen committed active notebooks retained top-level `metadata.papermill.output_path` values under `/tmp/*_out.ipynb`, making source notebooks look like smoke-output artifacts. | Fixed | Added a failing E14 verifier regression, implemented `E14.tmp_papermill_output_path`, stripped the stale top-level papermill metadata, and reran `python scripts/verify_repo.py --check execution --fast` to 0 findings. |
| OM-025 | Low | §3.32 examples/library-fit docs | `notebooks/self_supervised-fmnist-jepa-pytorch/README.md`, `notebooks/moe-fmnist-mixture-of-experts-pytorch/README.md` | Two README files still described the TinyShakespeare transformer notebook as part of the `NNDataset` default-batch workaround, despite its custom sequence dataset. | Fixed | Updated the wording to refer only to the diffusion, MoE, and JEPA `NNDataset` workaround set. |
| OM-026 | Medium | §3.31 consumed contracts | `docs/dependency-contracts.md`, `vendor/genai-vanilla` | The genai-vanilla ledger's observed upstream `main` SHA was stale after a new upstream commit; the JupyterHub service wording also implied the top-level compose file defined the service directly rather than including its fragment. | Fixed | Updated the observed upstream SHA to `b0bce0fc4e9d2bb282fbc6c97631f3e37233e24e`; verified pinned `c89eb5e7...` has `start.sh`, top-level `docker-compose.yml`, and `services/jupyterhub/compose.yml` defining `jupyterhub`. |
| OM-027 | Low | §3.31 consumed contracts | `scripts/start-jupyterhub.sh`, `deploy/genai-vanilla-jupyterhub.override.yml` | Wrapper comments said `vendor/genai-vanilla` was "pinned to main" even though the repository uses a pinned submodule snapshot that can intentionally lag upstream `main`. | Fixed | Reworded the wrapper and override comments to describe a pinned submodule snapshot; shellcheck and bash syntax validation cover the wrapper. |
| OM-028 | Low | §3.29 build reproducibility / §3.31 consumed contracts | `docs/dependency-contracts.md`, `Dockerfile` | The bootstrap tooling ledger said the Dockerfile installed `pip setuptools wheel`, but the Dockerfile currently upgrades only `pip` and `setuptools`. | Fixed | Corrected the ledger wording and kept the broader exact-pin hardening under OM-019. |
| OM-029 | Medium | §3.17 notebook execution / §3.32 examples-library fit | `notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb`, `tests/nnx_surface/test_notebook_api_surface.py` | The extended GAT notebook trained a single local `runs` list but ranked `NNRun.all()`, so committed "best run" outputs could show a cross-experiment `graph_sage` run instead of the notebook's GAT run. | Fixed | Added a failing static guard, changed the notebook to rank local `runs`, updated the markdown, cleared stale dependent outputs, and reran the focused guard plus notebook JSON parse. |
| OM-030 | Medium | §3.13 tests / §3.32 examples-library fit | `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`, `notebooks/quantization-mnist-ffnn-pytorch/` | The manual-only quantization notebook depended on `nnx.quantize_int8`, `nnx.qat_train_step_factory`, and `nnx.QATLifecycleCallback`, but no fast CI/surface guard covered those facade contracts after the notebook left papermill tiers. | Fixed | Added quantization facade signature checks plus backend-gated PTQ/QAT smoke tests that skip cleanly when `torchao` or `torch.int1` is unavailable; focused tests pass. |
