# scripts/docs/links.py
"""Markdown link extraction + three-surface self-containment rules (spec §7)."""

from __future__ import annotations

import re
from dataclasses import dataclass

REPO_URL = "github.com/thekaveh/ml-eng-lab"
WIKI_URL = "github.com/thekaveh/ml-eng-lab/wiki"
SITE_URL = "thekaveh.github.io/ml-eng-lab"

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Link:
    target: str


def find_links(md: str) -> list[Link]:
    return [Link(target=t) for t in _LINK_RE.findall(md)]


def is_forbidden(target: str, surface: str) -> bool:
    """True if `target` (from `surface`) points at a different surface or a GitHub file/source view."""
    t = target.strip()
    if not t.startswith(("http://", "https://")):
        return False  # relative links are intra-surface; integrity-checked separately
    on_site = SITE_URL in t
    on_wiki = WIKI_URL in t
    on_repo = (REPO_URL in t) and not on_wiki
    if surface == "site":
        return on_repo or on_wiki
    if surface == "wiki":
        return on_repo or on_site
    if surface == "repo":
        return on_site or on_wiki
    return False
