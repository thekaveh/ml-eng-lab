#!/usr/bin/env python3
"""CI gate: self-containment, completeness, placeholders, determinism (spec §6.3)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.docs.links import find_links, is_forbidden
from scripts.docs.manifest import Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    message: str


def check_self_containment(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for surface in ("site", "wiki"):
        d = generated_root / surface
        if not d.exists():
            continue
        for md in d.rglob("*.md"):
            for link in find_links(md.read_text(encoding="utf-8")):
                if is_forbidden(link.target, surface):
                    findings.append(Finding("error", f"{surface}: {md.relative_to(generated_root)} links cross-surface: {link.target}"))
    return findings


def check_completeness(manifest: Manifest, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for n in manifest.notebooks:
        if not (repo_root / n.spec).exists():
            findings.append(Finding("error", f"notebook spec missing: {n.spec}"))
        if not (repo_root / n.doc).exists():
            findings.append(Finding("error", f"notebook doc missing: {n.doc}"))
    return findings


def check_placeholders(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for md in generated_root.rglob("*.md"):
        if md.parent.name == "superpowers":
            continue
        for m in _PLACEHOLDER_RE.finditer(md.read_text(encoding="utf-8")):
            findings.append(Finding("error", f"placeholder {m.group(0)!r} in {md.relative_to(generated_root)}"))
    return findings


def check(repo_root: Path, generated_root: Path) -> int:
    from scripts.docs.build_docs import build

    manifest = load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    rc = build(repo_root / "docs/manifest.yaml", repo_root, check=True)
    if rc != 0:
        return rc
    findings: list[Finding] = []
    findings += check_self_containment(generated_root)
    findings += check_completeness(manifest, repo_root)
    findings += check_placeholders(generated_root)
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        print(f"[{f.severity.upper()}] {f.message}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return check(REPO_ROOT, REPO_ROOT / "generated")


if __name__ == "__main__":
    sys.exit(main())
