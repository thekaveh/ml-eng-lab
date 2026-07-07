# tests/test_push_wiki.py
from __future__ import annotations

from pathlib import Path

from scripts.docs.push_wiki import authenticated_remote, sync_wiki


def test_authenticated_remote_returns_ssh_command_pinning_key():
    cmd = authenticated_remote("git@github.com:thekaveh/ml-eng-lab.wiki.git", Path("/tmp/k"))
    assert cmd == "ssh -i /tmp/k -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"


def test_sync_wiki_copies_files_and_cleans_stale(tmp_path):
    src = tmp_path / "wiki"
    src.mkdir()
    (src / "Home.md").write_text("home", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Stale.md").write_text("old", encoding="utf-8")  # should be removed
    (repo / ".git").mkdir()  # must be preserved
    written = sync_wiki(src, repo)
    assert (repo / "Home.md").read_text() == "home"
    assert not (repo / "Stale.md").exists()
    assert (repo / ".git").exists()  # git metadata preserved
    assert written == [repo / "Home.md"]
