# 2.1 System & context view

This page describes the repository as a notebook-driven ML lab rather than as a deployable
service. The primary runtime objects are experiment directories under `notebooks/`, the notebook
execution tiers owned by the `Makefile`, the validation scripts under `scripts/`, and the three
documentation surfaces derived from the manifest-declared canonical source set. That set spans
the documentation tree under `docs/` and direct-root governance Markdown such as `SECURITY.md`.

## 2.1.1 System architecture

The system view establishes the ownership boundary: this repository owns notebook tasks,
consumer configuration, and verification, while NNx and Atlas remain separately versioned
upstream dependencies.

![ml-eng-lab system architecture](diagrams/img/system.png)

![ml-eng-lab runtime flow](diagrams/img/runtime-flow.png)

## 2.1.2 Three-surface documentation pipeline

![Three-surface documentation sync](diagrams/img/docs-sync.png)

![Documentation publishing flow](diagrams/img/docs-publishing.png)

The documentation system is the cleanest worked example of the lab's "canonical source +
derived surface" discipline. One manifest-declared canonical source set feeds three surfaces:

1. **Manifest.** `docs/manifest.yaml` declares the hierarchy, numbering, notebook specs, diagram
   masters, Markdown under `docs/`, and direct-root governance pages such as `SECURITY.md`. It is
   the single source of truth for the generated page set — if a page is not in the manifest, it
   does not appear in the generated site or wiki.
2. **Site generation.** `scripts/docs/build_docs.py --site` reads the manifest, applies the
   per-surface transforms in `scripts/docs/transforms.py` (forbidden-link stripping,
   path rewriting, image-asset rewriting), and writes `generated/site/` plus a generated
   `mkdocs.yml`. MkDocs Material then builds the published site from `generated/site/`.
3. **Wiki generation.** `scripts/docs/build_docs.py --wiki` applies the wiki surface transform
   (slugified numbered filenames, `Home.md` / `_Sidebar.md` / `_Footer.md` convention) and writes
   `generated/wiki/`. `scripts/docs/push_wiki.py` mirrors it to the GitHub wiki using a
   dedicated deploy key.
4. **Diagram rendering.** `scripts/docs/render_diagrams.py` rasterizes each manifest-declared
   HTML master to SVG (for the site) and PNG (committed under `docs/diagrams/img/` for the
   repository and wiki).
5. **CI gate.** `scripts/docs/check_docs.py` enforces self-containment (no cross-surface HTTP
   links), completeness (every manifest entry has a source file), placeholder-freeness, and
   generation determinism. It exits non-zero on any error.

The transforms guarantee that a link written once in a canonical source resolves correctly on
every surface: relative canonical paths are rewritten to per-surface output paths, image
references are rewritten to the surface-appropriate asset format, and any link that would cross
a surface boundary (a site page linking to a GitHub source view, for example) is stripped to
bare text.

The root `README.md` and `docs/index.md` openings are hand-authored. Their project
identity, active inventory and brand assets are checked without requiring equal prose;
it is not a manifest-generated page. The repository renders both that opener and the canonical
manifest sources directly, while the site and wiki contain only manifest-projected pages.

## 2.1.3 Runtime entry paths

A contributor opens the repository through one of four supported entry paths — local VS Code with
the Atlas JupyterHub kernel, a local venv, the Docker image, or GitHub Codespaces — and runs or
edits an experiment under `notebooks/<task>/`. The Atlas path uses the pinned `infra/` submodule,
the `ml-eng` track, and a parent-owned checkout mount; local VS Code remains the default editor.
Notebook-local `./data/` and `./runs/` paths resolve inside each experiment directory; `Makefile`
targets execute notebooks by changing into each notebook directory before invoking papermill, so
the task-local path invariant holds.

The validation path keeps Atlas parent policy distinct from direct submodule validation. The
unconditional `atlas-consumer-policy` job is intended to be a required gate on every pull request;
it ShellChecks the parent wrappers and runs `make test-atlas-consumer` without live services. The
path-scoped `atlas-contract` remains the non-required direct recursive-submodule validator for
declared Atlas inputs.

The unconditional `dependency-audit` job is an isolated `dependency-audit` signal: it runs
`make audit-advisories` against the reviewed JSON policy without starting Atlas or any service.
It remains separate from `pytest-repository` and `atlas-consumer-policy` so a live advisory-feed
failure is attributable and isolated. The final GitHub ruleset controller update makes it the
third required context alongside those existing two; that external setting is Task 5 work, not a
local documentation change.

`scripts/verify_repo.py`, the complete `make test` / `pytest-repository` contract, the focused NNx
and Ruff job, documentation checks, and notebook execution tiers retain their separate checks of
structure, documentation, library surfaces, and executable notebook behavior before changes are
merged.

Every local, CI, Docker, and Codespaces runtime enters through the four-stage canonical installer,
performs its last asset install, then freezes package state across pip-check, Torch verification,
NNx verification, and workload. No repository container starts Jupyter, Atlas, Ollama, or ComfyUI
as part of Issue #62.

`requirements/lock-policy.toml` is the machine-readable authority for human inputs, exact package
sources, three supported targets, allowed direct-URL/sdist exceptions, and generated locks. CI and
Codespaces select the matching hash-required lock; the Docker image creates
`/home/jovyan/.venvs/ml-eng-lab`, puts it first on `PATH`, and sets
`CONDA_AUTO_ACTIVATE_BASE=false` so the Jupyter startup hook cannot replace the reviewed runtime
with ambient conda packages. Image source references are exact tag-plus-index digests from
`requirements/image-lock.json`. Offline verification proves committed coherence; networked lock
and image checks prove external freshness.

## 2.1.4 Boundary decisions

- `notebooks/archive/` is preserved as read-only historical material and excluded from active
  notebook validation.
- `thekaveh-nnx[lm]==0.2.0` is consumed from PyPI; shared library changes land upstream in
  `thekaveh/NNx` before this repo bumps the pin.
- The quantization notebook is Tier B under Issue #66. Its bounded execution proves complete PTQ,
  QAT conversion, exact checkpoint reconstruction, and a machine-readable semantic result.
- `infra/` is a reviewed Atlas gitlink. Consumer configuration remains outside the submodule, and
  host-native Ollama is mandatory; a containerized Ollama service is not an approved runtime.
- Atlas CI preserves the ownership boundary: unconditional `atlas-consumer-policy` enforces the
  intended-required parent policy, while path-scoped, non-required `atlas-contract` validates the
  pinned submodule directly. Neither check starts or contacts Atlas, JupyterHub, Ollama, ComfyUI,
  Docker Compose, or unrelated containers.
- Dependency audit preserves the same boundary: its parent-owned Atlas contract does not initialize
  Atlas or start services. Issue #63 owns `requirements/lock-policy.toml` and the committed locks;
  Issue #65 still owns the Atlas runtime itself.
- The manifest-declared canonical source set under `docs/` and at approved root governance paths
  is the documentation source of truth; the generated site and wiki are never edited by hand.
