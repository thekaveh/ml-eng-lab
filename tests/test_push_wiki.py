# tests/test_push_wiki.py
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.docs.push_wiki import authenticated_remote, push_wiki, sync_wiki


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


def test_push_wiki_skips_commit_when_wiki_unchanged(tmp_path, monkeypatch):
    """No-op wiki sync must not crash on `git commit` (nothing staged)."""
    for k, v in {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(k, v)
    # local bare repo acting as the wiki remote (no network). GitHub wikis use `master`.
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "master", str(remote)], check=True)
    # seed the remote with an initial wiki state on master
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "init", "-b", "master", str(seed)], check=True)
    (seed / "Home.md").write_text("home", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(seed), "push", str(remote), "master"], check=True)
    # wiki src byte-identical to the live remote → nothing staged after `add -A`
    src = tmp_path / "wiki"
    src.mkdir()
    (src / "Home.md").write_text("home", encoding="utf-8")
    # without the guard, `git commit` exits 1 → CalledProcessError; guard returns 0
    rc = push_wiki(src, str(remote), key_path=None, push=True)
    assert rc == 0
