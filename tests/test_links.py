# tests/test_links.py
from __future__ import annotations

from scripts.docs.links import find_links, is_forbidden


def test_find_links_extracts_markdown_targets():
    md = "see [arch](architecture.md) and [repo](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md)"
    targets = [l.target for l in find_links(md)]
    assert targets == ["architecture.md", "https://github.com/thekaveh/ml-eng-lab/blob/main/README.md"]


def test_is_forbidden_relative_links_are_never_forbidden():
    assert is_forbidden("architecture.md", "site") is False
    assert is_forbidden("../notebooks/x/README.md", "wiki") is False


def test_is_forbidden_site_may_not_link_to_repo_or_wiki():
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/blob/main/README.md", "site") is True
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/wiki/Home", "site") is True
    # site -> site (same surface) is allowed
    assert is_forbidden("https://thekaveh.github.io/ml-eng-lab/architecture/", "site") is False


def test_is_forbidden_wiki_may_not_link_to_repo_or_site():
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/blob/main/x.md", "wiki") is True
    assert is_forbidden("https://thekaveh.github.io/ml-eng-lab/", "wiki") is True
    # wiki -> wiki (same surface) is allowed
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/wiki/Home", "wiki") is False


def test_is_forbidden_repo_may_not_link_to_site_or_wiki():
    assert is_forbidden("https://thekaveh.github.io/ml-eng-lab/", "repo") is True
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/wiki/Home", "repo") is True
    # repo -> repo (file view) is allowed (it IS the repo surface)
    assert is_forbidden("https://github.com/thekaveh/ml-eng-lab/blob/main/README.md", "repo") is False
