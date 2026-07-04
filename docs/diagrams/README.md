# Diagram Provenance

This directory contains checked-in standalone HTML architecture diagrams embedded by
`docs/architecture.md` and copied into the MkDocs `site/` build.

## 1. Current Artifacts

- `ml-eng-lab-system.html` describes the repository context and primary components.
- `ml-eng-lab-runtime-flow.html` describes supported runtime entry paths and task-local
  notebook artifact paths.
- `ml-eng-lab-notebook-sequence.html` traces notebook execution from parameters through
  training, ranking, visualization, persistence, and verification.
- `ml-eng-lab-docs-publishing.html` describes README, docs, MkDocs, GitHub Pages, wiki, and
  repository metadata surfaces.

## 2. Generation Contract

These diagrams were produced as dark, standalone HTML/SVG architecture artifacts using the
repository maintenance run's approved architecture-diagram workflow. The checked-in HTML files
are the source of truth for the current rendered site; do not hand-edit generated geometry
without updating this provenance note in the same change.

When a diagram needs to change:

1. Regenerate the affected HTML artifact with the same dark technical style, landscape
   layout, embedded SVG, and no external runtime JavaScript.
2. Confirm `docs/architecture.md` still embeds and links the artifact.
3. Run `make docs-build` and compare the generated `site/diagrams/*.html` copies to
   `docs/diagrams/*.html`.
4. Inspect the rendered diagram at normal browser zoom for overlapping labels, boxes,
   legends, or arrows before committing.

## 3. Review Rules

- Diagram content must describe the current repository behavior, not aspirational design.
- Arrows and labels should remain readable on desktop and narrow screens.
- New architecture, runtime, notebook-flow, or documentation-publishing changes should update
  the matching diagram in the same pull request.
