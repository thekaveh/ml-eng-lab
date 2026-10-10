#!/usr/bin/env python3
"""CI gate: self-containment, completeness, placeholders, determinism (spec §6.3)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from scripts.docs.links import SITE_URL, WIKI_URL, find_links, is_forbidden
from scripts.docs.manifest import Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HTML_H1_RE = re.compile(r'^<h1\s+align=["\']center["\']>(.+?)</h1>$', re.IGNORECASE)
_NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\.)?(?:\s+|$)")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+).*?\1")
_HTML_TAG_RE = re.compile(r"<(?P<closing>/)?(?P<tag>[A-Za-z][\w:-]*)\b[^>]*>")
# Stable union of CommonMark's block tag set and Python-Markdown's block elements.
_HTML_BLOCK_ELEMENTS = frozenset(
    {
        "address",
        "article",
        "aside",
        "base",
        "basefont",
        "blockquote",
        "body",
        "canvas",
        "caption",
        "center",
        "col",
        "colgroup",
        "dd",
        "details",
        "dialog",
        "dir",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "frame",
        "frameset",
        "group",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "iframe",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "math",
        "menu",
        "menuitem",
        "nav",
        "noframes",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "param",
        "pre",
        "progress",
        "script",
        "search",
        "section",
        "style",
        "summary",
        "table",
        "tbody",
        "td",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "title",
        "tr",
        "track",
        "ul",
        "video",
    }
)
_HTML_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_NUMBERED_H2_RE = re.compile(r"^##\s+\d+(?:\.\d)*(?:\.)?(?:\s+|$)", re.MULTILINE)
_PROJECT_SUMMARY_RE = re.compile(
    r"<!-- project-summary:start -->\s*(.*?)\s*<!-- project-summary:end -->",
    re.DOTALL,
)
PROJECT_TAGLINE = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts."
_PROJECT_POSTER_MARKUP = {
    "README.md": (
        '<p align="center">\n'
        '  <img src="docs/assets/ml-eng-lab-poster.png" '
        'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
        "</p>"
    ),
    "docs/index.md": (
        '<p align="center">\n'
        '  <img src="assets/ml-eng-lab-poster.png" '
        'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
        "</p>"
    ),
}
_PROJECT_CENTERED_TITLES = {
    "README.md": '<h1 align="center">ML ENG LAB</h1>',
    "docs/index.md": '<h1 align="center">1 · ML ENG LAB</h1>',
}
_PROJECT_ASSET_PREFIXES = {
    "README.md": "docs/assets/",
    "docs/index.md": "assets/",
}
PROJECT_BADGE_GROUPS = (
    (
        "Core ML",
        (
            ("Python", "python.svg"),
            ("Jupyter", "jupyter.svg"),
            ("NumPy", "numpy.svg"),
            ("pandas", "pandas.svg"),
            ("PyTorch", "pytorch.svg"),
            ("PyTorch Geometric", "pytorch-geometric.svg"),
            ("scikit-learn", "scikit-learn.svg"),
        ),
    ),
    (
        "NLP and graphs",
        (
            ("spaCy", "spacy.svg"),
            ("NLTK", "nltk.svg"),
            ("NetworkX", "networkx.svg"),
        ),
    ),
    (
        "Runtime",
        (
            ("Atlas", "atlas.svg"),
            ("Docker", "docker.svg"),
            ("VS Code", "vscode.svg"),
            ("GitHub Codespaces", "github-codespaces.svg"),
        ),
    ),
    (
        "Engineering",
        (
            ("NNx", "nnx.svg"),
            ("Papermill", "papermill.svg"),
            ("pytest", "pytest.svg"),
            ("Ruff", "ruff.svg"),
            ("GitHub Actions", "github-actions.svg"),
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    message: str


class _RawImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str | None] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "img":
            return
        self.sources.append(next((value for name, value in attrs if name == "src"), None))


def _is_fence_closer(line: str, marker: str, opening_length: int) -> bool:
    closing_match = re.match(
        r"^ {0,3}(" + re.escape(marker) + r"+)(?:[ \t]*)$",
        line,
    )
    return bool(
        closing_match and len(closing_match.group(1)) >= opening_length
    )


def _advance_html_container_stack(line: str, stack: list[str]) -> None:
    for match in _HTML_TAG_RE.finditer(line):
        tag = match.group("tag").lower()
        if not match.group("closing") and tag in _HTML_VOID_ELEMENTS:
            continue
        if match.group("closing"):
            if tag in stack:
                del stack[len(stack) - 1 - stack[::-1].index(tag) :]
        elif tag not in _HTML_VOID_ELEMENTS and not match.group(0).rstrip().endswith("/>"):
            stack.append(tag)


def _starts_html_container(line: str) -> bool:
    match = re.match(r"^ {0,3}<(?P<tag>[A-Za-z][\w:-]*)\b[^>]*>", line)
    return bool(
        match
        and match.group("tag").lower() not in _HTML_VOID_ELEMENTS
        and not match.group(0).rstrip().endswith("/>")
    )


def _rendered_markdown(text: str) -> str:
    rendered: list[str] = []
    fence: tuple[str, int] | None = None
    in_html_comment = False
    html_container_stack: list[str] = []
    for raw_line in text.splitlines():
        if fence is not None:
            if _is_fence_closer(raw_line, *fence):
                fence = None
            continue
        line, in_html_comment = _strip_html_comments(raw_line, in_html_comment)
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            opening = fence_match.group(1)
            fence = (opening[0], len(opening))
            continue
        if not line.strip():
            html_container_stack[:] = [
                tag for tag in html_container_stack if tag in _HTML_BLOCK_ELEMENTS
            ]
        if (
            not html_container_stack
            and not _starts_html_container(line)
            and (line.startswith("    ") or line.startswith("\t"))
        ):
            continue
        rendered_line = _INLINE_CODE_RE.sub("", line)
        rendered.append(rendered_line)
        _advance_html_container_stack(rendered_line, html_container_stack)
    return "\n".join(rendered)


def _raw_html_image_sources(text: str) -> list[str | None]:
    parser = _RawImageParser()
    parser.feed(_rendered_markdown(text))
    parser.close()
    return parser.sources


def check_self_containment(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for surface in ("site", "wiki"):
        d = generated_root / surface
        if not d.exists():
            continue
        surface_root = d.resolve()
        for md in d.rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            for link in find_links(text):
                if is_forbidden(link.target, surface):
                    findings.append(Finding("error", f"{surface}: {md.relative_to(generated_root)} links cross-surface: {link.target}"))
            for source in _raw_html_image_sources(text):
                if source is None or not source.strip():
                    findings.append(
                        Finding(
                            "error",
                            f"{surface}: {md.relative_to(generated_root)} raw HTML image missing src",
                        )
                    )
                    continue
                target = source.strip()
                parsed = urlsplit(target)
                if parsed.scheme.lower() in {"http", "https", "data"}:
                    continue
                if parsed.scheme:
                    findings.append(
                        Finding(
                            "error",
                            f"{surface}: {md.relative_to(generated_root)} unsupported raw HTML image source: {target}",
                        )
                    )
                    continue
                if parsed.netloc:
                    continue
                image_path = unquote(parsed.path)
                candidate = (
                    d / image_path.lstrip("/")
                    if image_path.startswith("/")
                    else md.parent / image_path
                )
                resolved = candidate.resolve()
                if not resolved.is_relative_to(surface_root):
                    findings.append(
                        Finding(
                            "error",
                            f"{surface}: {md.relative_to(generated_root)} local image target escapes generated surface: {target}",
                        )
                    )
                    continue
                if not resolved.is_file():
                    findings.append(
                        Finding(
                            "error",
                            f"{surface}: {md.relative_to(generated_root)} local image target missing: {target}",
                        )
                    )
    return findings


def check_repo_self_containment(
    repo_root: Path,
    manifest: Manifest | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    if manifest is None and (repo_root / "docs/manifest.yaml").exists():
        manifest = load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    candidates = {repo_root / "README.md", *(repo_root / "docs").rglob("*.md")}
    if manifest is not None:
        candidates.update(
            repo_root / source
            for source in manifest_markdown_sources(manifest)
        )
    for md in sorted(candidates):
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for target in (SITE_URL, WIKI_URL):
            if target in text:
                findings.append(
                    Finding(
                        "error",
                        f"repo: {md.relative_to(repo_root)} references another documentation surface: {target}",
                    )
                )
    return findings


def manifest_markdown_sources(manifest: Manifest) -> set[str]:
    sources: set[str] = set()

    def visit(section) -> None:
        if section.source:
            sources.add(section.source)
        for child in section.children:
            visit(child)

    for section in manifest.sections:
        visit(section)
    sources.update(notebook.doc for notebook in manifest.notebooks)
    return sources


def _manifest_page_numbers(manifest: Manifest) -> dict[str, str]:
    numbers: dict[str, str] = {}

    def visit(section) -> None:
        if section.source:
            numbers[section.source] = section.number
        for child in section.children:
            visit(child)

    for section in manifest.sections:
        visit(section)
    numbers.update((notebook.doc, notebook.number) for notebook in manifest.notebooks)
    return numbers


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None
    in_html_comment = False
    for raw_line in text.splitlines():
        if fence is not None:
            marker, opening_length = fence
            if _is_fence_closer(raw_line, marker, opening_length):
                fence = None
            continue

        line, in_html_comment = _strip_html_comments(raw_line, in_html_comment)
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            opening = fence_match.group(1)
            fence = (opening[0], len(opening))
            continue
        html_match = _HTML_H1_RE.match(line)
        if html_match:
            headings.append((1, html_match.group(1)))
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            headings.append((len(heading_match.group(1)), heading_match.group(2)))
    return headings


def _strip_html_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    while True:
        if in_comment:
            end = line.find("-->")
            if end == -1:
                return "", True
            line = line[end + 3 :]
            in_comment = False

        start = line.find("<!--")
        if start == -1:
            return line, False
        end = line.find("-->", start + 4)
        if end == -1:
            return line[:start], True
        line = line[:start] + line[end + 3 :]


def check_numbering(manifest: Manifest, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for source, page_number in sorted(_manifest_page_numbers(manifest).items()):
        path = repo_root / source
        if not path.exists():
            continue
        headings = _markdown_headings(path.read_text(encoding="utf-8"))
        if not headings or headings[0][0] != 1:
            findings.append(Finding("error", f"{source}: missing H1 for manifest number {page_number}"))
            continue

        last_number: dict[int, str] = {}
        sibling_count: dict[int, int] = {}
        h1_count = 0
        for level, title in headings:
            number_match = _NUMBER_PREFIX_RE.match(title)
            actual = number_match.group(1) if number_match else None
            if level == 1:
                h1_count += 1
                if h1_count > 1:
                    findings.append(Finding("error", f"{source}: multiple H1 headings are not allowed"))
                expected = page_number
            else:
                parent = last_number.get(level - 1)
                if parent is None:
                    findings.append(Finding("error", f"{source}: H{level} has no H{level - 1} parent: {title}"))
                    continue
                sibling_count[level] = sibling_count.get(level, 0) + 1
                expected = f"{parent}.{sibling_count[level]}"

            if actual != expected:
                findings.append(
                    Finding(
                        "error",
                        f"{source}: H{level} must start with {expected} (got {title!r})",
                    )
                )
            last_number[level] = expected
            for deeper in range(level + 1, 7):
                last_number.pop(deeper, None)
                sibling_count.pop(deeper, None)
    return findings


def check_completeness(manifest: Manifest, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    declared = {
        source
        for source in manifest_markdown_sources(manifest)
        if source.startswith("docs/")
    }
    actual = {
        str(path.relative_to(repo_root))
        for path in (repo_root / "docs").rglob("*.md")
    }
    for source in sorted(declared - actual):
        findings.append(Finding("error", f"manifest Markdown source missing: {source}"))
    for source in sorted(actual - declared):
        findings.append(Finding("error", f"canonical Markdown not declared in manifest: {source}"))
    for n in manifest.notebooks:
        if not (repo_root / n.spec).exists():
            findings.append(Finding("error", f"notebook spec missing: {n.spec}"))
    return findings


def check_notebook_infrastructure(manifest: Manifest, repo_root: Path) -> list[Finding]:
    from scripts.docs.notebook_infrastructure import (
        NotebookInfrastructureError,
        load_atlas_task_contracts,
        render_atlas_task_table,
        verify_atlas_task_table,
    )

    try:
        contracts = load_atlas_task_contracts(repo_root, manifest)
        verify_atlas_task_table(
            repo_root / "docs/notebook-infrastructure.md",
            render_atlas_task_table(contracts),
        )
    except (NotebookInfrastructureError, OSError) as error:
        return [Finding("error", f"notebook infrastructure: {error}")]
    return []


def check_placeholders(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for md in generated_root.rglob("*.md"):
        if md.parent.name == "superpowers":
            continue
        for m in _PLACEHOLDER_RE.finditer(md.read_text(encoding="utf-8")):
            findings.append(Finding("error", f"placeholder {m.group(0)!r} in {md.relative_to(generated_root)}"))
    return findings


def _normalize_prose(text: str) -> str:
    return " ".join(text.split())


def _badge_rows(asset_prefix: str) -> str:
    rows: list[str] = []
    for label, badges in PROJECT_BADGE_GROUPS:
        images = " ".join(
            f'<img alt="{alt}" src="{asset_prefix}badges/{filename}">'
            for alt, filename in badges
        )
        rows.append(
            '<p align="center">\n'
            f"  <sub><strong>{label}</strong></sub><br>\n"
            f"  {images}\n"
            "</p>"
        )
    return "\n\n".join(rows)


def check_project_opening(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    expected_tagline = (
        f'<p align="center"><strong>{PROJECT_TAGLINE}</strong></p>'
    )

    for relative_path, poster in _PROJECT_POSTER_MARKUP.items():
        path = repo_root / relative_path
        if not path.exists():
            findings.append(Finding("error", f"project opener source missing: {relative_path}"))
            continue
        text = path.read_text(encoding="utf-8")
        centered_title = _PROJECT_CENTERED_TITLES[relative_path]
        asset_prefix = _PROJECT_ASSET_PREFIXES[relative_path]
        expected_prefix = (
            f"{poster}\n\n"
            f"{centered_title}\n\n"
            f"{expected_tagline}\n\n"
            f"{_badge_rows(asset_prefix)}\n\n"
            "<!-- project-summary:start -->\n"
        )
        if not text.startswith(expected_prefix):
            findings.append(
                Finding(
                    "error",
                    f"project opener order or structure is invalid in {relative_path}",
                )
            )
        matches = _PROJECT_SUMMARY_RE.findall(text)
        if len(matches) != 1:
            findings.append(
                Finding("error", f"project summary markers must occur exactly once in {relative_path}")
            )
            continue
        summary_match = _PROJECT_SUMMARY_RE.search(text)
        assert summary_match is not None
        following_summary = text[summary_match.end() :]
        h2_match = _NUMBERED_H2_RE.search(following_summary)
        if h2_match is None:
            findings.append(
                Finding(
                    "error",
                    f"project summary must be followed by a numbered H2 in {relative_path}",
                )
            )
            opener = text
            opener_with_tail = text
        else:
            summary_tail = following_summary[: h2_match.start()]
            opener = text[: summary_match.end()]
            opener_with_tail = text[: summary_match.end() + h2_match.start()]
            if summary_tail.strip():
                findings.append(
                    Finding(
                        "error",
                        f"project opener tail after summary must contain only whitespace in {relative_path}",
                    )
                )
        if "runtime-flow" in opener_with_tail:
            findings.append(
                Finding(
                    "error",
                    f"runtime-flow diagram cannot appear in project opener in {relative_path}",
                )
            )
        if centered_title not in opener:
            findings.append(
                Finding("error", f"centered HTML title missing from {relative_path}"))
        if poster not in opener:
            findings.append(Finding("error", f"project opener poster missing from {relative_path}"))
        if expected_tagline not in opener:
            findings.append(Finding("error", f"canonical project tagline missing from {relative_path}"))

        poster_source = re.search(r'<img src="([^"]+)"', poster)
        assert poster_source is not None
        resolved_poster = Path(relative_path).parent / poster_source.group(1)
        if not (repo_root / resolved_poster).is_file():
            findings.append(
                Finding("error", f"project poster asset missing: {resolved_poster}")
            )

        for _, badges in PROJECT_BADGE_GROUPS:
            for alt, filename in badges:
                source = f"{asset_prefix}badges/{filename}"
                badge = f'<img alt="{alt}" src="{source}">'
                if badge not in opener:
                    findings.append(
                        Finding(
                            "error",
                            f"project badge {alt!r} missing from {relative_path}",
                        )
                    )
                resolved_source = Path(relative_path).parent / source
                if not (repo_root / resolved_source).is_file():
                    findings.append(
                        Finding(
                            "error",
                            f"project badge asset missing: {resolved_source}",
                        )
                    )
        summary = _normalize_prose(matches[0])
        for pattern, label in (
            (r"\bml-eng-lab\b", "project identity"),
            (r"\bnotebook", "notebook purpose"),
            (r"\b(?:21|twenty-one)\s+active task (?:folders|directories)", "active task inventory"),
            (r"\b(?:29|twenty-nine)\s+(?:active )?notebooks", "active notebook inventory"),
        ):
            if not re.search(pattern, summary, re.IGNORECASE):
                findings.append(Finding("error", f"project summary missing {label} in {relative_path}"))

    return findings


def check(repo_root: Path, generated_root: Path) -> int:
    from scripts.docs.build_docs import build

    manifest = load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    findings = check_notebook_infrastructure(manifest, repo_root)
    if findings:
        for finding in findings:
            print(f"[{finding.severity.upper()}] {finding.message}", file=sys.stderr)
        return 1 if any(finding.severity == "error" for finding in findings) else 0
    rc = build(repo_root / "docs/manifest.yaml", repo_root, check=True)
    if rc != 0:
        return rc
    findings = []
    findings += check_self_containment(generated_root)
    findings += check_repo_self_containment(repo_root, manifest)
    findings += check_completeness(manifest, repo_root)
    findings += check_numbering(manifest, repo_root)
    findings += check_placeholders(generated_root)
    findings += check_project_opening(repo_root)
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        print(f"[{f.severity.upper()}] {f.message}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return check(REPO_ROOT, REPO_ROOT / "generated")


if __name__ == "__main__":
    sys.exit(main())
