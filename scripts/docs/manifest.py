"""Canonical documentation manifest: hierarchy, numbering, notebook specs, diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ManifestError(ValueError):
    """Raised when the manifest is malformed or references missing files."""


@dataclass(frozen=True)
class Section:
    id: str
    number: str
    title: str
    source: str | None = None
    children: list["Section"] = field(default_factory=list)
    diagrams: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NotebookEntry:
    task: str
    number: str
    family: str
    depth: str
    doc: str
    spec: str
    diagrams: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiagramEntry:
    id: str
    master: str


@dataclass(frozen=True)
class Manifest:
    surfaces: tuple[str, ...]
    numbering: str
    sections: list[Section]
    notebooks: list[NotebookEntry]
    diagrams: list[DiagramEntry]


def _section(data: dict) -> Section:
    return Section(
        id=str(data["id"]),
        number=str(data["number"]),
        title=str(data["title"]),
        source=data.get("source"),
        children=[_section(c) for c in data.get("children", [])],
        diagrams=list(data.get("diagrams", [])),
    )


def parse_manifest(text: str) -> Manifest:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ManifestError(f"manifest is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a YAML mapping")
    try:
        numbering = data.get("numbering")
        if numbering != "baked":
            raise ManifestError(f"numbering must be 'baked' (got {numbering!r})")
        surfaces = tuple(data.get("surfaces", []))
        if surfaces != ("repo", "site", "wiki"):
            raise ManifestError(f"surfaces must be [repo, site, wiki] (got {list(surfaces)})")
        return Manifest(
            surfaces=surfaces,
            numbering=str(numbering),
            sections=[_section(s) for s in data.get("sections", [])],
            notebooks=[
                NotebookEntry(
                    task=str(n["task"]),
                    number=str(n["number"]),
                    family=str(n["family"]),
                    depth=str(n["depth"]),
                    doc=str(n["doc"]),
                    spec=str(n["spec"]),
                    diagrams=list(n.get("diagrams", [])),
                )
                for n in data.get("notebooks", [])
            ],
            diagrams=[DiagramEntry(id=str(d["id"]), master=str(d["master"])) for d in data.get("diagrams", [])],
        )
    except KeyError as e:
        raise ManifestError(f"manifest entry missing required key: {e}") from e


def _check_exists(repo_root: Path, rel: str, kind: str) -> None:
    if not (repo_root / rel).exists():
        raise ManifestError(f"{kind} {rel!r} does not exist (resolved under {repo_root})")


def load_manifest(path: Path, repo_root: Path) -> Manifest:
    manifest = parse_manifest(path.read_text(encoding="utf-8"))
    for s in manifest.sections:
        if s.source:
            _check_exists(repo_root, s.source, "section source")
        for c in s.children:
            if c.source:
                _check_exists(repo_root, c.source, "section source")
    for n in manifest.notebooks:
        _check_exists(repo_root, n.doc, "notebook doc")
        _check_exists(repo_root, n.spec, "notebook spec")
    for d in manifest.diagrams:
        _check_exists(repo_root, d.master, "diagram master")
    return manifest
