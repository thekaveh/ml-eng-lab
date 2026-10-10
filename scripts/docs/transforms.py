# scripts/docs/transforms.py
"""Per-surface markdown transforms: strip forbidden links, rewrite paths (spec §7)."""

from __future__ import annotations

import re
import posixpath
from urllib.parse import urlsplit, unquote

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


def rewrite_for_surface(
    md: str, surface: str, source_map: dict[str, str], *, source_path: str = ""
) -> str:
    """Resolve links at their canonical page, then relative to the rendered page."""
    def repl(m: re.Match[str]) -> str:
        text, target = m.group(1), m.group(2).strip()
        if is_forbidden(target, surface):
            return text
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return m.group(0)
        path = unquote(parsed.path)
        source_relative = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), path))
        key = next((candidate for candidate in (source_relative, path) if candidate in source_map), None)
        if key is not None:
            destination = source_map[key]
            if surface == "wiki":
                destination = destination.removesuffix(".md")
            elif source_path in source_map:
                destination = posixpath.relpath(destination, posixpath.dirname(source_map[source_path]) or ".")
            suffix = ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
            return f"[{text}]({destination}{suffix})"
        if path.endswith((".md", ".ipynb")):
            return text
        return m.group(0)

    return _LINK_RE.sub(repl, md)
