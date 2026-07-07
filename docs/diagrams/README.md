# Diagram Provenance

This directory holds the diagram **masters**: standalone, dark, HTML/SVG architecture
artifacts produced by the architecture-diagram skill. Each master is the single source
for one diagram; the docs pipeline derives per-surface artifacts from it so all three
documentation surfaces embed the same geometry.

## 1. Pipeline

Each diagram declared in `docs/manifest.yaml` (`diagrams[].master` → a file in this
directory) is rendered once into two artifacts by `scripts/docs/render_diagrams.py` and
`scripts/docs/build_docs.py`:

- **SVG** → `generated/site/assets/img/<id>.svg` (crisp, themeable) — embedded by the
  generated `.io` site. `render_site` extracts the inline `<svg>` from the master, so the
  site output is complete and deterministic.
- **PNG** → `docs/diagrams/img/<id>.png` (committed) — embedded by the in-repo markdown
  and the generated wiki. `render_diagrams.py` rasterizes the SVG via `cairosvg`.

Because the same master feeds all three surfaces, diagrams never drift: updating a master
and re-running the pipeline refreshes every surface.

## 2. Current Artifacts

- `ml-eng-lab-system.html` — repository context and primary components (declared as
  `system` in the manifest; embedded by `docs/architecture.md`).
- `ml-eng-lab-runtime-flow.html` — runtime entry paths and task-local notebook artifact
  paths (added to the manifest later).
- `ml-eng-lab-notebook-sequence.html` — notebook execution from parameters through
  training, ranking, visualization, persistence, and verification (planned).
- `ml-eng-lab-docs-publishing.html` — README, docs, MkDocs, GitHub Pages, wiki, and
  repository metadata surfaces (planned).
- `ml-eng-lab-docs-sync.html` — the three-surface documentation sync pipeline (added with
  the docs-overhaul foundation).

## 3. Generation Contract

The checked-in HTML masters are the source of truth. Do not hand-edit generated geometry
without updating this provenance note in the same change. When a diagram needs to change:

1. Regenerate the affected HTML master with the same dark technical style, landscape
   layout, embedded SVG, and no external runtime JavaScript.
2. Run `python -m scripts.docs.render_diagrams` to refresh the committed PNG
   (`docs/diagrams/img/<id>.png`); the site SVG is refreshed by the next `build_docs` run.
3. Run `make docs-build` (render + `build_docs --site` + `mkdocs build --strict`) and
   confirm both surfaces embed the updated geometry.
4. Inspect the rendered diagram at normal browser zoom for overlapping labels, boxes,
   legends, or arrows before committing.

## 4. Review Rules

- Diagram content must describe current repository behavior, not aspirational design.
- Arrows and labels should remain readable on desktop and narrow screens.
- New architecture, runtime, notebook-flow, or documentation-publishing changes should
  update the matching master and add or refresh its manifest entry in the same pull request.
- A diagram referenced by the manifest must have its master present, or
  `scripts/docs/manifest.py:load_manifest` and `render_diagrams.py` will fail.
