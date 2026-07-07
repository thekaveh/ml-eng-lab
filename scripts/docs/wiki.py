"""Generate the GitHub-wiki surface (generated/wiki/) from canonical docs."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.docs.manifest import Manifest
from scripts.docs.transforms import build_source_map, rewrite_for_surface

_PNG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*diagrams/img/([^.]+)\.png\)")


def _rewrite_images_wiki(md: str) -> str:
    return _PNG_RE.sub(lambda m: f"![{m.group(1)}](img/{m.group(2)}.png)", md)


def render_wiki(manifest: Manifest, repo_root: Path, out_dir: Path) -> list[Path]:
    source_map = build_source_map(manifest, "wiki")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(src_rel: str) -> Path:
        text = (repo_root / src_rel).read_text(encoding="utf-8")
        text = rewrite_for_surface(text, "wiki", source_map)
        text = _rewrite_images_wiki(text)
        dest = out_dir / source_map[src_rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
        return dest

    home = None
    for s in manifest.sections:
        if s.source:
            dest = emit(s.source)
            if s.id == "overview":
                home = dest
        for c in s.children:
            if c.source:
                emit(c.source)
    for n in manifest.notebooks:
        emit(n.doc)

    # GitHub wiki convention: Home.md is the landing page.
    if home and home.name != "Home.md":
        home.rename(out_dir / "Home.md")

    # Sidebar (numbered nav) + footer.
    sidebar = ["# ml-eng-lab wiki", ""]
    for s in manifest.sections:
        label = f"{s.number}. {s.title}"
        if s.source and s.id != "overview":
            sidebar.append(f"- [{label}]({Path(source_map[s.source]).stem})")
        for c in s.children:
            if c.source:
                sidebar.append(f"  - [{c.number}. {c.title}]({Path(source_map[c.source]).stem})")
    if manifest.notebooks:
        sidebar.append("- [8. Notebooks]")
        for n in manifest.notebooks:
            sidebar.append(f"  - [{n.number}. {n.task}]({Path(source_map[n.doc]).stem})")
    (out_dir / "_Sidebar.md").write_text("\n".join(sidebar) + "\n", encoding="utf-8")
    (out_dir / "_Footer.md").write_text("Self-contained ml-eng-lab wiki.\n", encoding="utf-8")

    # copy PNG assets
    img_out = out_dir / "img"
    for d in manifest.diagrams:
        png = repo_root / "docs/diagrams/img" / f"{d.id}.png"
        if png.exists():
            img_out.mkdir(parents=True, exist_ok=True)
            (img_out / png.name).write_bytes(png.read_bytes())
    return written
