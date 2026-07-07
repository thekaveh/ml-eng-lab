# tests/test_render_diagrams.py
from __future__ import annotations

import pytest

cairosvg = pytest.importorskip("cairosvg")  # skips the whole module if cairosvg absent

from scripts.docs.render_diagrams import extract_svg, render_all, svg_to_png  # noqa: E402
from scripts.docs.manifest import parse_manifest  # noqa: E402

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"><rect width="100" height="50" fill="#00e5ff"/></svg>'
HTML = f"<html><body>{SVG}</body></html>"


def test_extract_svg_pulls_inline_svg():
    assert extract_svg(HTML) == SVG


def test_svg_to_png_writes_png(tmp_path):
    out = tmp_path / "d.png"
    svg_to_png(SVG, out, width=200)
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_all_writes_svg_and_png(tmp_path):
    masters = tmp_path / "docs/diagrams"
    masters.mkdir(parents=True)
    (masters / "ml-eng-lab-system.html").write_text(HTML, encoding="utf-8")
    site_img = tmp_path / "generated/site/assets/img"
    png_dir = tmp_path / "docs/diagrams/img"
    manifest = parse_manifest(
        """
surfaces: [repo, site, wiki]
numbering: baked
sections: []
notebooks: []
diagrams:
  - id: system
    master: docs/diagrams/ml-eng-lab-system.html
"""
    )
    written = render_all(manifest, tmp_path, site_img, png_dir, width=200)
    assert (site_img / "system.svg").exists()
    assert (png_dir / "system.png").exists()
    assert written == [tmp_path / "docs/diagrams/ml-eng-lab-system.html"]
