#!/usr/bin/env python3
"""Generate the .io site input (generated/site/) + root mkdocs.yml from canonical docs."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from scripts.docs.manifest import Manifest, load_manifest
from scripts.docs.project_assets import (
    cleanup_generated_output,
    copy_project_assets,
    validate_generated_output,
)
from scripts.docs.transforms import build_source_map, rewrite_for_surface

REPO_ROOT = Path(__file__).resolve().parents[2]

# Image refs: in-repo PNG path → site SVG asset path. Preserve any ../ prefix so images
# resolve from subdirectory docs (e.g. docs/notebooks/<task>.md → ../assets/img/<id>.svg).
_PNG_RE = re.compile(r"!\[([^\]]*)\]\(((?:\.\./)*)diagrams/img/([^.]+)\.png\)")


def _rewrite_images_site(md: str) -> str:
    return _PNG_RE.sub(lambda m: f"![{m.group(1)}]({m.group(2)}assets/img/{m.group(3)}.svg)", md)


def render_site(
    manifest: Manifest,
    repo_root: Path,
    out_dir: Path,
    *,
    trusted_output_root: Path,
) -> list[Path]:
    validate_generated_output(trusted_output_root, out_dir)
    source_map = build_source_map(manifest, "site")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    expected: set[Path] = set()

    def emit(src_rel: str) -> None:
        text = (repo_root / src_rel).read_text(encoding="utf-8")
        text = rewrite_for_surface(text, "site", source_map, source_path=src_rel)
        text = _rewrite_images_site(text)
        dest = out_dir / source_map[src_rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
        expected.add(dest)

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
        css_dest = out_dir / "stylesheets/extra.css"
        css_dest.write_text(css_src.read_text(encoding="utf-8"), encoding="utf-8")
        expected.add(css_dest)
    # copy mathjax loader (referenced by generated mkdocs.yml)
    js_src = repo_root / "docs/javascripts/mathjax.js"
    if js_src.exists():
        (out_dir / "javascripts").mkdir(parents=True, exist_ok=True)
        js_dest = out_dir / "javascripts/mathjax.js"
        js_dest.write_text(js_src.read_text(encoding="utf-8"), encoding="utf-8")
        expected.add(js_dest)
    # place diagram SVGs (crisp, for the site) — extracted from committed HTML masters,
    # so render_site owns the complete, deterministic site output.
    from scripts.docs.render_diagrams import extract_svg

    assets = out_dir / "assets/img"
    for d in manifest.diagrams:
        svg = extract_svg((repo_root / d.master).read_text(encoding="utf-8"))
        assets.mkdir(parents=True, exist_ok=True)
        svg_dest = assets / f"{d.id}.svg"
        svg_dest.write_text(svg, encoding="utf-8")
        expected.add(svg_dest)
    copy_project_assets(
        repo_root,
        out_dir,
        expected,
        trusted_output_root=trusted_output_root,
    )
    cleanup_generated_output(
        out_dir,
        expected,
        trusted_output_root=trusted_output_root,
    )
    return written


def _nav_lines(manifest: Manifest) -> list[str]:
    lines: list[str] = ["nav:"]
    navigation = [(int(section.number.split(".")[0]), "section", section) for section in manifest.sections]
    if manifest.notebooks:
        notebook_number = int(manifest.notebooks[0].number.split(".")[0])
        navigation.append((notebook_number, "notebooks", None))
    for _, kind, s in sorted(navigation, key=lambda item: item[0]):
        if kind == "notebooks":
            lines.append('  - "8. Notebooks":')
            for n in manifest.notebooks:
                lines.append(f'      - "{n.number}. {n.task}": {n.doc.removeprefix("docs/")}')
            continue
        assert s is not None
        if s.source and s.children:
            lines.append(f'  - "{s.number}. {s.title}":')
            lines.append(f'      - "{s.number}. {s.title}": {s.source.removeprefix("docs/")}')
            for c in s.children:
                if c.source:
                    lines.append(f'      - "{c.number}. {c.title}": {c.source.removeprefix("docs/")}')
        elif s.source:
            lines.append(f'  - "{s.number}. {s.title}": {s.source.removeprefix("docs/")}')
        elif s.children:
            lines.append(f'  - "{s.number}. {s.title}":')
            for c in s.children:
                if c.source:
                    lines.append(f'      - "{c.number}. {c.title}": {c.source.removeprefix("docs/")}')
    return lines


_MKDOCS_TEMPLATE = """\
site_name: ml-eng-lab
site_description: Self-contained machine-learning notebook experiments with reproducible local runtimes.
site_url: https://thekaveh.github.io/ml-eng-lab/
docs_dir: generated/site
site_dir: site
use_directory_urls: true
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
        render_site(
            manifest,
            repo_root,
            out_dir,
            trusted_output_root=repo_root,
        )
        (repo_root / "mkdocs.yml").write_text(render_mkdocs_yml(manifest, repo_root, out_dir), encoding="utf-8")
    if wiki or check:
        from scripts.docs.wiki import render_wiki  # lazy; keeps `mkdocs`-less checks lightweight

        render_wiki(
            manifest,
            repo_root,
            repo_root / "generated/wiki",
            trusted_output_root=repo_root,
        )
    if check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            temporary_root = Path(td)
            render_site(
                manifest,
                repo_root,
                temporary_root / "site",
                trusted_output_root=temporary_root,
            )
            _assert_dirs_equal(
                temporary_root / "site",
                repo_root / "generated/site",
                a_trusted_output_root=temporary_root,
                b_trusted_output_root=repo_root,
            )
            render_wiki(
                manifest,
                repo_root,
                temporary_root / "wiki",
                trusted_output_root=temporary_root,
            )
            _assert_dirs_equal(
                temporary_root / "wiki",
                repo_root / "generated/wiki",
                a_trusted_output_root=temporary_root,
                b_trusted_output_root=repo_root,
            )
    return 0


def _assert_dirs_equal(
    a: Path,
    b: Path,
    *,
    a_trusted_output_root: Path,
    b_trusted_output_root: Path,
) -> None:
    validate_generated_output(a_trusted_output_root, a)
    validate_generated_output(b_trusted_output_root, b)

    def snapshot(d: Path) -> dict[str, tuple[str, str]]:
        entries: dict[str, tuple[str, str]] = {}
        for path in d.rglob("*"):
            relative = path.relative_to(d).as_posix()
            if path.is_symlink():
                entries[relative] = ("symlink", str(path.readlink()))
            elif path.is_dir():
                entries[relative] = ("directory", "")
            elif path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                entries[relative] = ("file", digest)
            else:
                entries[relative] = ("other", "")
        return entries

    a_snap, b_snap = snapshot(a), snapshot(b)
    if a_snap == b_snap:
        return
    only_a = sorted(set(a_snap) - set(b_snap))
    only_b = sorted(set(b_snap) - set(a_snap))
    type_diff = sorted(
        path
        for path in a_snap
        if path in b_snap and a_snap[path][0] != b_snap[path][0]
    )
    content_diff = sorted(
        path
        for path in a_snap
        if path in b_snap
        and a_snap[path][0] == b_snap[path][0]
        and a_snap[path][1] != b_snap[path][1]
    )
    raise AssertionError(
        "generation not deterministic: "
        f"only-in-temp={only_a}, only-in-generated={only_b}, "
        f"type-diff={type_diff}, content-diff={content_diff}"
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
