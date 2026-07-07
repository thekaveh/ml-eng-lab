#!/usr/bin/env python3
"""Generate the .io site input (generated/site/) + root mkdocs.yml from canonical docs."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from scripts.docs.manifest import Manifest, load_manifest
from scripts.docs.transforms import build_source_map, rewrite_for_surface

REPO_ROOT = Path(__file__).resolve().parents[2]

# Image refs: in-repo PNG path → site SVG asset path. Preserve any ../ prefix so images
# resolve from subdirectory docs (e.g. docs/notebooks/<task>.md → ../assets/img/<id>.svg).
_PNG_RE = re.compile(r"!\[([^\]]*)\]\(((?:\.\./)*)diagrams/img/([^.]+)\.png\)")


def _rewrite_images_site(md: str) -> str:
    return _PNG_RE.sub(lambda m: f"![{m.group(1)}]({m.group(2)}assets/img/{m.group(3)}.svg)", md)


def render_site(manifest: Manifest, repo_root: Path, out_dir: Path) -> list[Path]:
    source_map = build_source_map(manifest, "site")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(src_rel: str) -> None:
        text = (repo_root / src_rel).read_text(encoding="utf-8")
        text = rewrite_for_surface(text, "site", source_map)
        text = _rewrite_images_site(text)
        dest = out_dir / source_map[src_rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)

    for s in manifest.sections:
        if s.source:
            emit(s.source)
        for c in s.children:
            if c.source:
                emit(c.source)
    for n in manifest.notebooks:
        emit(n.doc)

    # copy theme stylesheet
    css_src = repo_root / "docs/stylesheets/extra.css"
    if css_src.exists():
        (out_dir / "stylesheets").mkdir(parents=True, exist_ok=True)
        (out_dir / "stylesheets/extra.css").write_text(css_src.read_text(encoding="utf-8"), encoding="utf-8")
    # copy mathjax loader (referenced by generated mkdocs.yml)
    js_src = repo_root / "docs/javascripts/mathjax.js"
    if js_src.exists():
        (out_dir / "javascripts").mkdir(parents=True, exist_ok=True)
        (out_dir / "javascripts/mathjax.js").write_text(js_src.read_text(encoding="utf-8"), encoding="utf-8")
    # place diagram SVGs (crisp, for the site) — extracted from committed HTML masters,
    # so render_site owns the complete, deterministic site output.
    from scripts.docs.render_diagrams import extract_svg

    assets = out_dir / "assets/img"
    for d in manifest.diagrams:
        svg = extract_svg((repo_root / d.master).read_text(encoding="utf-8"))
        assets.mkdir(parents=True, exist_ok=True)
        (assets / f"{d.id}.svg").write_text(svg, encoding="utf-8")
    return written


def _nav_lines(manifest: Manifest) -> list[str]:
    lines: list[str] = ["nav:"]
    for s in manifest.sections:
        if s.source:
            lines.append(f'  - "{s.number}. {s.title}": {s.source.removeprefix("docs/")}')
        elif s.children:
            lines.append(f'  - "{s.number}. {s.title}":')
            for c in s.children:
                if c.source:
                    lines.append(f'      - "{c.number}. {c.title}": {c.source.removeprefix("docs/")}')
    if manifest.notebooks:
        lines.append('  - "8. Notebooks":')
        for n in manifest.notebooks:
            lines.append(f'      - "{n.number}. {n.task}": {n.doc.removeprefix("docs/")}')
    return lines


_MKDOCS_TEMPLATE = """\
site_name: ml-eng-lab
site_description: Self-contained machine-learning notebook experiments with reproducible local runtimes.
site_url: https://thekaveh.github.io/ml-eng-lab/
docs_dir: generated/site
site_dir: site
use_directory_urls: true
exclude_docs: |
  superpowers/**
# No repository URL / name / edit URI keys — surfaces are fully self-contained (spec D2).
theme:
  name: material
  language: en
  palette:
    - scheme: slate
      primary: cyan
      accent: cyan
      toggle:
        icon: material/weather-sunny
        name: Switch to light mode
    - scheme: default
      primary: cyan
      accent: cyan
      toggle:
        icon: material/weather-night
        name: Switch to dark mode
  font:
    text: Inter
    code: JetBrains Mono
  features:
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - toc.follow
    - content.code.copy
    - content.code.annotate
    - content.tooltips
    - header.autohide
extra_css:
  - stylesheets/extra.css
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - footnotes
  - def_list
  - pymdownx.superfences
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.keys
  - pymdownx.arithmatex:
      generic: true
  - toc:
      permalink: true
extra_javascript:
  - javascripts/mathjax.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
{nav}
"""


def render_mkdocs_yml(manifest: Manifest, repo_root: Path, site_dir: Path) -> str:
    nav = "\n".join(_nav_lines(manifest))
    return _MKDOCS_TEMPLATE.format(nav=nav)


def build(manifest_path: Path, repo_root: Path, *, site: bool = False, wiki: bool = False, check: bool = False) -> int:
    manifest = load_manifest(manifest_path, repo_root)
    if site or check:
        out_dir = repo_root / "generated/site"
        render_site(manifest, repo_root, out_dir)
        (repo_root / "mkdocs.yml").write_text(render_mkdocs_yml(manifest, repo_root, out_dir), encoding="utf-8")
    if wiki or check:
        from scripts.docs.wiki import render_wiki  # lazy; keeps `mkdocs`-less checks lightweight

        render_wiki(manifest, repo_root, repo_root / "generated/wiki")
    if check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            render_site(manifest, repo_root, Path(td) / "site")
            _assert_dirs_equal(Path(td) / "site", repo_root / "generated/site")
            render_wiki(manifest, repo_root, Path(td) / "wiki")
            _assert_dirs_equal(Path(td) / "wiki", repo_root / "generated/wiki")
    return 0


def _assert_dirs_equal(a: Path, b: Path) -> None:
    def snapshot(d: Path) -> dict[str, str]:
        return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob("*") if p.is_file()}

    a_snap, b_snap = snapshot(a), snapshot(b)
    if a_snap == b_snap:
        return
    only_a = sorted(set(a_snap) - set(b_snap))
    only_b = sorted(set(b_snap) - set(a_snap))
    content_diff = sorted(p for p in a_snap if p in b_snap and a_snap[p] != b_snap[p])
    raise AssertionError(
        f"generation not deterministic: only-in-temp={only_a}, only-in-generated={only_b}, content-diff={content_diff}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site", action="store_true")
    p.add_argument("--wiki", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    if not (args.site or args.wiki or args.check):
        p.error("specify at least one of --site / --wiki / --check")
    return build(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT, site=args.site, wiki=args.wiki, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
