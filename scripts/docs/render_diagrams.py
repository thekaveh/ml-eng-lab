#!/usr/bin/env python3
"""Render diagram HTML masters → SVG (site) + PNG (committed, in-repo + wiki).

Each master is a standalone HTML file with an inline <svg> (architecture-diagram
skill). We extract the SVG and rasterize via cairosvg. No browser dependency.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
# Named HTML entities that are NOT predefined in XML (lt/gt/amp/quot/apos) break cairosvg +
# browsers loading the standalone .svg. Convert them to their unicode characters.
_NON_XML_ENTITY_RE = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#)[a-zA-Z]+;")


def extract_svg(html_src: str) -> str:
    match = SVG_RE.search(html_src)
    if not match:
        raise ValueError("no inline <svg> found in diagram HTML master")
    return _NON_XML_ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), match.group(0))


def svg_to_png(svg: str, out_path: Path, *, width: int) -> None:
    import cairosvg  # lazy import — keeps the module importable without cairosvg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_path), output_width=width)


def render_all(manifest, repo_root: Path, site_img_dir: Path, png_dir: Path, *, width: int = 1600) -> list[Path]:
    written: list[Path] = []
    for d in manifest.diagrams:
        master = repo_root / d.master
        html = master.read_text(encoding="utf-8")
        svg = extract_svg(html)
        site_img_dir.mkdir(parents=True, exist_ok=True)
        (site_img_dir / f"{d.id}.svg").write_text(svg, encoding="utf-8")
        svg_to_png(svg, png_dir / f"{d.id}.png", width=width)
        written.append(master)
    return written


def main(argv: list[str] | None = None) -> int:
    from scripts.docs.manifest import load_manifest

    repo_root = REPO_ROOT
    manifest = load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    render_all(
        manifest,
        repo_root,
        site_img_dir=repo_root / "generated/site/assets/img",
        png_dir=repo_root / "docs/diagrams/img",
    )
    print(f"rendered {len(manifest.diagrams)} diagram(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
