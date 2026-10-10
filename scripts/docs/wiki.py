"""Generate the GitHub-wiki surface (generated/wiki/) from canonical docs."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.docs.manifest import Manifest
from scripts.docs.project_assets import (
    cleanup_generated_output,
    copy_project_assets,
    validate_generated_output,
)
from scripts.docs.transforms import build_source_map, rewrite_for_surface

_PNG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*diagrams/img/([^.]+)\.png\)")


def _rewrite_images_wiki(md: str) -> str:
    return _PNG_RE.sub(lambda m: f"![{m.group(1)}](img/{m.group(2)}.png)", md)


def render_wiki(
    manifest: Manifest,
    repo_root: Path,
    out_dir: Path,
    *,
    trusted_output_root: Path,
) -> list[Path]:
    validate_generated_output(trusted_output_root, out_dir)
    source_map = build_source_map(manifest, "wiki")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    expected: set[Path] = set()

    def emit(src_rel: str) -> Path:
        text = (repo_root / src_rel).read_text(encoding="utf-8")
        text = rewrite_for_surface(text, "wiki", source_map, source_path=src_rel)
        text = _rewrite_images_wiki(text)
        dest = out_dir / source_map[src_rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
        expected.add(dest)
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
        home_dest = out_dir / "Home.md"
        home.rename(home_dest)
        expected.remove(home)
        expected.add(home_dest)

    # Sidebar (numbered nav) + footer.
    sidebar = ["# ml-eng-lab wiki", ""]
    navigation = [(int(section.number.split(".")[0]), "section", section) for section in manifest.sections]
    if manifest.notebooks:
        notebook_number = int(manifest.notebooks[0].number.split(".")[0])
        navigation.append((notebook_number, "notebooks", None))
    for _, kind, s in sorted(navigation, key=lambda item: item[0]):
        if kind == "notebooks":
            prefix = manifest.notebooks[0].number.split(".")[0]
            sidebar.append(f"- {prefix}. Notebooks")
            for n in manifest.notebooks:
                sidebar.append(f"  - [{n.number}. {n.task}]({Path(source_map[n.doc]).stem})")
            continue
        assert s is not None
        label = f"{s.number}. {s.title}"
        if s.source and s.id != "overview":
            sidebar.append(f"- [{label}]({Path(source_map[s.source]).stem})")
        else:
            sidebar.append(f"- {label}")
        for c in s.children:
            if c.source:
                sidebar.append(f"  - [{c.number}. {c.title}]({Path(source_map[c.source]).stem})")
    sidebar_dest = out_dir / "_Sidebar.md"
    footer_dest = out_dir / "_Footer.md"
    sidebar_dest.write_text("\n".join(sidebar) + "\n", encoding="utf-8")
    footer_dest.write_text("Self-contained ml-eng-lab wiki.\n", encoding="utf-8")
    expected.update((sidebar_dest, footer_dest))

    # copy PNG assets
    img_out = out_dir / "img"
    for d in manifest.diagrams:
        png = repo_root / "docs/diagrams/img" / f"{d.id}.png"
        if png.exists():
            img_out.mkdir(parents=True, exist_ok=True)
            png_dest = img_out / png.name
            png_dest.write_bytes(png.read_bytes())
            expected.add(png_dest)
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
