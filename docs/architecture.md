# 1. Architecture

This page describes the repository as a notebook-driven ML lab rather than as a deployable
service. The primary runtime objects are experiment directories, notebook execution tiers,
validation scripts, and documentation surfaces.

Diagram provenance and regeneration rules are tracked in
[Diagram Provenance](diagrams/README.md).

## 1.1. System Context

The diagram below is generated as a standalone HTML architecture artifact and embedded into
the documentation site.

<iframe class="architecture-frame" src="../diagrams/ml-eng-lab-system.html" title="ml-eng-lab system architecture"></iframe>

[Open the diagram in a full page](diagrams/ml-eng-lab-system.html).

## 1.2. Runtime Flow

The runtime flow diagram shows the supported entry paths and the invariant that notebook code
resolves `./data` and `./runs` from the task directory.

<iframe class="architecture-frame" src="../diagrams/ml-eng-lab-runtime-flow.html" title="ml-eng-lab runtime flow"></iframe>

[Open the runtime flow diagram in a full page](diagrams/ml-eng-lab-runtime-flow.html).

1. A contributor opens the repository through a local venv, Docker image, Codespaces, or the
   vendored genai-vanilla JupyterHub stack.
2. They run or edit an experiment under `notebooks/<task>/`.
3. Notebook-local `./data/` and `./runs/` paths resolve inside that experiment directory.
4. `Makefile` targets execute notebooks by changing into each notebook directory before
   invoking papermill.
5. `scripts/verify_repo.py`, pytest, ruff, and CI verify structure, documentation, and public
   notebook surfaces before changes are merged.
6. MkDocs builds this documentation site from checked-in Markdown and publishes it through
   GitHub Pages.

## 1.3. Notebook Execution Sequence

The execution sequence diagram traces a task notebook from papermill parameters through
training, ranking, visualization, run persistence, and output-path verification.

<iframe class="architecture-frame" src="../diagrams/ml-eng-lab-notebook-sequence.html" title="ml-eng-lab notebook execution sequence"></iframe>

[Open the notebook execution sequence diagram in a full page](diagrams/ml-eng-lab-notebook-sequence.html).

## 1.4. Documentation Publishing

The publishing diagram describes the canonical documentation sources, generated site, wiki
signpost, and repository metadata surfaces.

<iframe class="architecture-frame" src="../diagrams/ml-eng-lab-docs-publishing.html" title="ml-eng-lab documentation publishing"></iframe>

[Open the documentation publishing diagram in a full page](diagrams/ml-eng-lab-docs-publishing.html).

## 1.5. Boundary Decisions

- `notebooks/archive/` is preserved as read-only historical material and excluded from active
  notebook validation.
- `thekaveh-nnx[lm]==0.2.0` is consumed from PyPI; shared library changes land upstream in
  `thekaveh/NNx` before this repo bumps the pin.
- The quantization notebook is active but manual-only until the pinned Torch stack can satisfy
  `torchao>=0.17`.
