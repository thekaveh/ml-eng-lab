#!/usr/bin/env python3
"""Push generated/wiki/ → ml-eng-lab.wiki.git via a Deploy Key (spec §10)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = "git@github.com:thekaveh/ml-eng-lab.wiki.git"


def authenticated_remote(remote: str, key_path: Path) -> str:
    """SSH command pinning the deploy key (used as GIT_SSH_COMMAND for wiki clone/push).

    `remote` is accepted for interface stability / future per-remote key selection.
    """
    return f"ssh -i {key_path} -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"


def sync_wiki(src: Path, repo_dir: Path) -> list[Path]:
    """Copy src/* into repo_dir/, preserving .git, removing stale files."""
    written: list[Path] = []
    keep = {".git"}
    for existing in repo_dir.iterdir():
        if existing.name not in keep and existing.is_dir():
            shutil.rmtree(existing)
        elif existing.name not in keep:
            existing.unlink()
    for item in src.iterdir():
        dest = repo_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        written.append(dest)
    return written


def push_wiki(src: Path, remote: str, key_path: Path | None, *, push: bool) -> int:
    if not src.exists():
        print(f"wiki source not found: {src}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        repo_dir = Path(td) / "wiki"
        cmd = ["git", "clone", "--depth", "1", remote, str(repo_dir)] if push else ["git", "init", str(repo_dir)]
        env = os.environ.copy()
        if key_path is not None:
            env["GIT_SSH_COMMAND"] = authenticated_remote(remote, key_path)
        subprocess.run(cmd, env=env, check=True)
        sync_wiki(src, repo_dir)
        if not push:
            print(f"wiki check ok ({len(list(src.iterdir()))} entries)")
            return 0
        subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], env=env, check=True)
        staged = subprocess.run(["git", "-C", str(repo_dir), "diff", "--cached", "--quiet"], env=env)
        if staged.returncode != 0:
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "docs: sync wiki"], env=env, check=True)
        # GitHub wikis render from the `master` branch (not the repo's default `main`);
        # `git clone` checks out `master`, so push local `master` → remote `master`.
        subprocess.run(["git", "-C", str(repo_dir), "push", remote, "master"], env=env, check=True)
        print("wiki pushed")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--remote", default=os.environ.get("WIKI_REMOTE", DEFAULT_REMOTE))
    args = p.parse_args(argv)
    if not (args.check or args.push):
        p.error("specify --check or --push")
    key = os.environ.get("WIKI_DEPLOY_KEY")
    key_path = Path(key) if key else None
    return push_wiki(REPO_ROOT / "generated/wiki", args.remote, key_path, push=args.push)


if __name__ == "__main__":
    sys.exit(main())
