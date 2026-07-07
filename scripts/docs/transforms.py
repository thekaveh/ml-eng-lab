# scripts/docs/transforms.py
"""Per-surface markdown transforms: strip forbidden links, rewrite paths (spec §7)."""

from __future__ import annotations

import re

from scripts.docs.links import is_forbidden
from scripts.docs.manifest import Manifest, NotebookEntry

_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _slugify(title: str) -> str:
    """Title → GitHub-wiki filename slug (e.g. 'System & context view' → 'System-context-view')."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "-", title).strip("-")
    return cleaned or "page"


def _wiki_name(number: str, title: str) -> str:
    num = number.replace(".", "-")
    return f"{num}-{_slugify(title)}.md"


def _section_output(section, surface: str) -> str:
    if surface == "wiki":
        return "Home.md" if section.id == "overview" else _wiki_name(section.number, section.title)
    return "index.md" if section.id == "overview" else section.source.removeprefix("docs/")


def _notebook_output(n: NotebookEntry, surface: str) -> str:
    if surface == "wiki":
        return _wiki_name(n.number, n.task)
    return n.doc.removeprefix("docs/")


def build_source_map(manifest: Manifest, surface: str) -> dict[str, str]:
    """Map canonical source path → output path for a surface."""
    sm: dict[str, str] = {}
    for s in manifest.sections:
        if s.source:
            sm[s.source] = _section_output(s, surface)
        for c in s.children:
            if c.source:
                sm[c.source] = _section_output(c, surface)
    for n in manifest.notebooks:
        sm[n.doc] = _notebook_output(n, surface)
    return sm


def rewrite_for_surface(md: str, surface: str, source_map: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        text, target = m.group(1), m.group(2).strip()
        if is_forbidden(target, surface):
            return text  # strip the link, keep the bare text
        if target.endswith(".ipynb"):
            return text  # notebooks have no page in the surface; drop to bare text
        if target in source_map:
            return f"[{text}]({source_map[target]})"
        return m.group(0)

    return _LINK_RE.sub(repl, md)
