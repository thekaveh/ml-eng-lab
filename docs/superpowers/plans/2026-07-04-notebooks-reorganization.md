# Notebooks Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every active and archived experiment into `notebooks/`, make each notebook rerunnable from its new directory, and rename live repository identity references from `ml-lab` to `ml-eng-lab`.

**Architecture:** Treat `notebooks/<experiment>/` as the canonical experiment root. Move active task directories wholesale under `notebooks/`, move archived CodeXGLUE experiments under `notebooks/archive/`, update verifier/tooling to understand the new root, then rewrite live docs and links. Preserve historical output/changelog text unless it is active guidance or a live path contract.

**Tech Stack:** Git moves, Python 3.11, nbformat, PyYAML, Makefile, GitHub Actions YAML, pytest, ruff, papermill.

## Global Constraints

- Active notebooks must live under `notebooks/<task>/`.
- Archived CodeXGLUE notebooks must live under `notebooks/archive/codexglue_summarization/`.
- Per-experiment `README.md`, task-local helper `.py` files, tracked archived `src/`, and tracked archived `model/` files move with their notebooks.
- Each notebook must be rerunnable from its new directory using local relative `./data`, `./runs`, `./model`, and sibling imports.
- Live repository identity must be `ml-eng-lab`, including GitHub URLs, nbviewer URLs, Docker tags, Codespaces examples, and JupyterHub bind-mount examples.
- Historical changelog entries, findings, and preserved notebook outputs may keep `ml-lab` only when they describe past events or recorded execution output.
- Do not redesign ML experiments or change model behavior.
- Do not re-execute expensive Tier-B or Tier-C notebooks in place.
- Use `git mv` for tracked moves whenever possible.
- Keep commits frequent and focused.

---

## File Structure

### New Canonical Layout

- `notebooks/<active-task>/README.md`
- `notebooks/<active-task>/*.ipynb`
- `notebooks/<active-task>/*.py` for task-local helper modules
- `notebooks/<active-task>/data/` ignored local data artifacts
- `notebooks/<active-task>/runs/` ignored local run artifacts
- `notebooks/archive/README.md`
- `notebooks/archive/codexglue_summarization/.gitignore`
- `notebooks/archive/codexglue_summarization/<experiment>/notebook.ipynb`
- `notebooks/archive/codexglue_summarization/<experiment>/src/`
- `notebooks/archive/codexglue_summarization/<experiment>/model/`

### Existing Files Modified

- `scripts/verify_repo.py`: notebook root helpers, active notebook discovery, numbered-doc scanning, Tier-C baseline lookup.
- `scripts/verify_repo_config.yaml`: active task slugs remain slugs; all notebook paths gain `notebooks/`.
- `tests/test_verify_repo.py`: expected paths and synthetic fixtures for verifier behavior.
- `tests/nnx_surface/test_notebook_api_surface.py`: active notebook discovery excludes `notebooks/archive/`.
- `tests/nnx_surface/*.py`: docstrings and path references that mention old notebook paths.
- `Makefile`: Tier notebook paths and comments.
- `.github/workflows/ci.yml`: Docker tag, artifact paths, comments.
- `pyproject.toml`: ruff ignore path for Tier-C notebooks.
- `.devcontainer/devcontainer.json`: display name, comments, repo path examples.
- `Dockerfile`: only live repository-name references found during implementation.
- `deploy/genai-vanilla-jupyterhub.override.yml`: bind mount target path if it names `ml-lab`.
- `scripts/start-jupyterhub.sh`: comments or mount-path wording that names `ml-lab`.
- `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/*.md`, per-experiment READMEs, notebook markdown cells.

### No Permanent New Runtime Code

Use one-off Python snippets for mechanical path rewrites and notebook markdown rewrites. Do not commit those snippets as permanent scripts.

---

### Task 1: Baseline Inventory And Safety Checks

**Files:**
- Read: `docs/superpowers/specs/2026-07-04-notebooks-reorganization-design.md`
- Read: `scripts/verify_repo_config.yaml`
- Read: `.gitignore`
- Read: `Makefile`
- Read: `.github/workflows/ci.yml`
- No committed file changes.

**Interfaces:**
- Consumes: approved design spec.
- Produces: `/tmp/ml-eng-lab-notebook-migration-map.tsv`, a tab-separated old-to-new path map used by later tasks.

- [ ] **Step 1: Confirm clean working tree**

Run:

```bash
git status --short --branch
```

Expected:

```text
## codex/overnight-maintenance
```

- [ ] **Step 2: Generate old-to-new notebook map**

Run:

```bash
python - <<'PY'
from pathlib import Path

repo = Path.cwd()
active = [
    p for p in sorted(repo.glob("*.ipynb"))
]
active = []
for p in sorted(repo.glob("*/*.ipynb")):
    if p.parts[0] == "archive":
        continue
    active.append(p)
archive = sorted((repo / "archive" / "codexglue_summarization").glob("*/*.ipynb"))

rows = []
for p in active:
    rows.append((str(p), str(Path("notebooks") / p)))
for p in archive:
    rows.append((str(p), str(Path("notebooks") / p)))

out = Path("/tmp/ml-eng-lab-notebook-migration-map.tsv")
out.write_text("\n".join(f"{old}\t{new}" for old, new in rows) + "\n", encoding="utf-8")
print(f"active={len(active)} archive={len(archive)} total={len(rows)}")
print(out)
PY
```

Expected:

```text
active=29 archive=22 total=51
/tmp/ml-eng-lab-notebook-migration-map.tsv
```

- [ ] **Step 3: Inventory ignored runtime artifacts**

Run:

```bash
find . -path './.git' -prune -o \( -type d -name data -o -type d -name runs -o -type d -name model \) -print | sort > /tmp/ml-eng-lab-artifact-dirs.txt
sed -n '1,200p' /tmp/ml-eng-lab-artifact-dirs.txt
```

Expected: output includes task-local `data/` and `runs/`, archive `model/`, and root `./runs`.

- [ ] **Step 4: Record root `runs/` ownership**

Run:

```bash
for d in runs/*; do
  [ -d "$d" ] || continue
  printf '== %s ==\n' "$d"
  sed -n '1,80p' "$d/run.yaml" 2>/dev/null || true
done
```

Expected:

- `runs/6b77c92e3fba5fce09fc2f3cf9df0fba` has `net: transformer`, `vocab_size: 256`, and belongs to `text_generation-tinyshakespeare-transformer-pytorch`.
- `runs/8da7cb3514e2157dddd5f65385487e72` has `net: transformer`, `vocab_size: 52`, and belongs to `preference_alignment-toy-dpo-pytorch`.
- `runs/best` is a symlink or directory alias inside root `runs/`; inspect it before moving.

- [ ] **Step 5: Run current fast baseline**

Run:

```bash
python scripts/verify_repo.py --check all --fast
pytest tests/test_verify_repo.py -v
pytest tests/nnx_surface/test_notebook_api_surface.py -v
ruff check . --no-cache
```

Expected:

- `verify_repo.py --check all --fast` exits 0, warnings allowed.
- Focused pytest commands pass.
- `ruff check . --no-cache` passes.

- [ ] **Step 6: Commit checkpoint only if baseline metadata was committed**

No files should be modified. If `git status --short` is clean, do not commit.

---

### Task 2: Update Verifier For `notebooks/` Root

**Files:**
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**
- Consumes: `ACTIVE_TASK_DIRS` slugs from `scripts/verify_repo_config.yaml`.
- Produces:
  - `NOTEBOOK_ROOT = Path("notebooks")`
  - `ARCHIVE_NOTEBOOK_ROOT = NOTEBOOK_ROOT / "archive"`
  - `_active_task_path(repo: Path, task: str) -> Path`
  - `_notebook_rel(path: Path, repo: Path) -> str`
  - `_baseline_notebook_rel(rel: str) -> str`

- [ ] **Step 1: Write failing tests for active notebook discovery under `notebooks/`**

Modify `tests/test_verify_repo.py` by adding this test near the existing D1 notebook tests:

```python
def test_iter_notebooks_reads_active_tasks_under_notebooks(tmp_path, monkeypatch):
    repo = tmp_path
    active = repo / "notebooks" / "task-a"
    archive = repo / "notebooks" / "archive" / "old-task"
    old_root = repo / "task-a"
    active.mkdir(parents=True)
    archive.mkdir(parents=True)
    old_root.mkdir()

    (active / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (archive / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (old_root / "notebook.ipynb").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("task-a",))

    found = [str(p.relative_to(repo)) for p in verify_repo._iter_notebooks(repo)]

    assert found == ["notebooks/task-a/notebook.ipynb"]
```

- [ ] **Step 2: Write failing test for Tier-C baseline path mapping**

Add this test near existing phase3 baseline tests:

```python
def test_baseline_notebook_rel_removes_notebooks_prefix():
    assert (
        verify_repo._baseline_notebook_rel(
            "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb"
        )
        == "node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb"
    )
    assert verify_repo._baseline_notebook_rel("legacy/notebook.ipynb") == "legacy/notebook.ipynb"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_verify_repo.py::test_iter_notebooks_reads_active_tasks_under_notebooks tests/test_verify_repo.py::test_baseline_notebook_rel_removes_notebooks_prefix -v
```

Expected: both tests fail because `_iter_notebooks` still scans root task dirs and `_baseline_notebook_rel` does not exist.

- [ ] **Step 4: Implement verifier path helpers**

Modify `scripts/verify_repo.py` near `ACTIVE_TASK_DIRS`:

```python
NOTEBOOK_ROOT = Path("notebooks")
ARCHIVE_NOTEBOOK_ROOT = NOTEBOOK_ROOT / "archive"


def _active_task_path(repo: Path, task: str) -> Path:
    return repo / NOTEBOOK_ROOT / task


def _baseline_notebook_rel(rel: str) -> str:
    prefix = f"{NOTEBOOK_ROOT.as_posix()}/"
    return rel.removeprefix(prefix)
```

Replace `_iter_notebooks` with:

```python
def _iter_notebooks(repo: Path) -> Iterator[Path]:
    for d in ACTIVE_TASK_DIRS:
        for nb_path in _active_task_path(repo, d).glob("*.ipynb"):
            yield nb_path
```

Replace active README/path loops in `_iter_in_scope_text_files`, `_iter_numbered_doc_files`, and `check_docs` so each uses `_active_task_path(repo, d) / "README.md"` instead of `repo / d / "README.md"`.

Replace `_phase3_code_cells_unchanged` phase3 glob:

```python
phase3 = list(_active_task_path(repo, "node_classification-reddit-gnn-pyg").glob("phase3-*.ipynb"))
```

Replace the baseline `git show` argument in `_phase3_code_cells_unchanged`:

```python
rel = str(nb.relative_to(repo))
baseline_rel = _baseline_notebook_rel(rel)
rc, raw, err = _run(["git", "show", f"pre-cleanup-baseline:{baseline_rel}"], repo)
```

Keep `location=rel` for findings so errors point at the new path.

- [ ] **Step 5: Run focused verifier tests**

Run:

```bash
pytest tests/test_verify_repo.py::test_iter_notebooks_reads_active_tasks_under_notebooks tests/test_verify_repo.py::test_baseline_notebook_rel_removes_notebooks_prefix -v
```

Expected: both tests pass.

- [ ] **Step 6: Run full verifier tests**

Run:

```bash
pytest tests/test_verify_repo.py -v
```

Expected: tests pass after any existing hardcoded path expectations are updated to include `notebooks/`.

- [ ] **Step 7: Commit**

Run:

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "test: teach verifier about notebooks root"
```

---

### Task 3: Update NNx Surface Notebook Discovery

**Files:**
- Modify: `tests/nnx_surface/test_notebook_api_surface.py`
- Modify: `tests/nnx_surface/test_image_classification_mnist_ffnn_pytorch.py`
- Modify: `tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py`
- Modify: `tests/nnx_surface/test_tabular_classification_iris_mlp_pytorch.py`
- Modify: `tests/nnx_surface/__init__.py`

**Interfaces:**
- Consumes: git-tracked notebook paths.
- Produces: active notebook discovery that includes `notebooks/<task>/*.ipynb` and excludes `notebooks/archive/**`.

- [ ] **Step 1: Update synthetic discovery test**

In `tests/nnx_surface/test_notebook_api_surface.py`, update `test_active_notebooks_uses_git_tracked_files` so the synthetic tracked paths are:

```python
tracked_nb = tmp_path / "notebooks" / "task" / "notebook.ipynb"
archive_nb = tmp_path / "notebooks" / "archive" / "old" / "notebook.ipynb"
checkpoint_nb = tmp_path / "notebooks" / "task" / ".ipynb_checkpoints" / "scratch.ipynb"
untracked_nb = tmp_path / "notebooks" / "task" / "scratch.ipynb"
```

And the fake `git ls-files` result is:

```python
return subprocess.CompletedProcess(
    cmd,
    0,
    stdout="\n".join(
        [
            "notebooks/task/notebook.ipynb",
            "notebooks/archive/old/notebook.ipynb",
            "notebooks/task/.ipynb_checkpoints/scratch.ipynb",
        ]
    )
    + "\n",
    stderr="",
)
```

The assertion should be:

```python
assert _active_notebooks(tmp_path) == [tracked_nb]
```

- [ ] **Step 2: Update `_active_notebooks` archive exclusion**

In `tests/nnx_surface/test_notebook_api_surface.py`, update `_active_notebooks` to exclude the new archive root:

```python
def _active_notebooks(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Every git-tracked notebook except archived notebooks and checkpoints."""
    proc = subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    notebooks = []
    for rel in proc.stdout.splitlines():
        parts = Path(rel).parts
        if parts[:2] == ("notebooks", "archive"):
            continue
        if ".ipynb_checkpoints" in parts:
            continue
        notebooks.append(repo_root / rel)
    return notebooks
```

- [ ] **Step 3: Update live docstrings and path strings**

Replace current path references in `tests/nnx_surface/*.py` from:

```text
image_classification-mnist-ffnn-pytorch/notebook.ipynb
node_classification-reddit-gnn-pyg/...
tabular_classification-iris-mlp-pytorch/notebook.ipynb
```

to:

```text
notebooks/image_classification-mnist-ffnn-pytorch/notebook.ipynb
notebooks/node_classification-reddit-gnn-pyg/...
notebooks/tabular_classification-iris-mlp-pytorch/notebook.ipynb
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/nnx_surface/test_notebook_api_surface.py::test_active_notebooks_uses_git_tracked_files -v
pytest tests/nnx_surface -v
```

Expected: tests pass before the move because synthetic behavior is independent and real discovery still sees current notebooks until Task 4 changes paths. If the real discovery count fails before the move, delay the full `pytest tests/nnx_surface -v` command until after Task 4, and run only the synthetic test now.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/nnx_surface
git commit -m "test: discover notebooks under notebooks root"
```

---

### Task 4: Move Active And Archived Experiment Trees

**Files:**
- Move: all active root task directories to `notebooks/<task>/`
- Move: `archive/README.md` to `notebooks/archive/README.md`
- Move: `archive/codexglue_summarization/` to `notebooks/archive/codexglue_summarization/`
- Move ignored artifacts by filesystem rename as part of directory moves.

**Interfaces:**
- Consumes: current root task directories and `archive/codexglue_summarization`.
- Produces: canonical `notebooks/` tree.

- [ ] **Step 1: Create destination root**

Run:

```bash
mkdir -p notebooks
```

Expected: `notebooks/` exists and is empty or contains only files created by an earlier aborted attempt. If it contains experiment directories, stop and inspect before continuing.

- [ ] **Step 2: Move active task directories**

Run:

```bash
python - <<'PY'
from pathlib import Path
import subprocess

active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]

root = Path("notebooks")
root.mkdir(exist_ok=True)
for task in active:
    src = Path(task)
    dst = root / task
    if not src.exists():
        raise SystemExit(f"missing source {src}")
    if dst.exists():
        raise SystemExit(f"destination exists {dst}")
    subprocess.run(["git", "mv", str(src), str(dst)], check=True)
PY
```

Expected: `git status --short` shows renames from root task directories to `notebooks/<task>/`.

- [ ] **Step 3: Move archive docs and experiments**

Run:

```bash
mkdir -p notebooks/archive
git mv archive/README.md notebooks/archive/README.md
git mv archive/codexglue_summarization notebooks/archive/codexglue_summarization
rmdir archive
```

Expected: `archive/` is removed. If `rmdir archive` reports the directory is not empty, run `find archive -maxdepth 2 -print` and move the remaining archive-owned files under `notebooks/archive/` before removing it.

- [ ] **Step 4: Move root run artifacts to owning experiments**

Run:

```bash
mkdir -p notebooks/text_generation-tinyshakespeare-transformer-pytorch/runs
mkdir -p notebooks/preference_alignment-toy-dpo-pytorch/runs

mv runs/6b77c92e3fba5fce09fc2f3cf9df0fba notebooks/text_generation-tinyshakespeare-transformer-pytorch/runs/
mv runs/8da7cb3514e2157dddd5f65385487e72 notebooks/preference_alignment-toy-dpo-pytorch/runs/

if [ -e runs/best ]; then
  printf 'root runs/best remains after owned run moves; inspect before deleting:\n'
  ls -la runs/best
fi

find runs -mindepth 1 -maxdepth 1 -print
```

Expected: either `runs/` is empty or only `runs/best` remains. If `runs/best` is a dangling symlink to one of the moved runs, remove it with `rm runs/best`. If `runs/best` is a real directory, inspect its `run.yaml` and move it to the owning experiment. Remove empty `runs/` with `rmdir runs`.

- [ ] **Step 5: Verify moved notebook counts**

Run:

```bash
find notebooks -path '*/.ipynb_checkpoints' -prune -o -name '*.ipynb' -print | sort > /tmp/notebooks-after.txt
awk 'BEGIN{active=0; archive=0} /^notebooks\/archive\//{archive++; next} /^notebooks\//{active++} END{printf "active=%d archive=%d total=%d\n", active, archive, active+archive}' /tmp/notebooks-after.txt
```

Expected:

```text
active=29 archive=22 total=51
```

- [ ] **Step 6: Verify old active and archive notebook paths are gone**

Run:

```bash
git ls-files '*.ipynb' | awk '$0 !~ /^notebooks\// {print}'
```

Expected: no output.

- [ ] **Step 7: Commit**

Run:

```bash
git add -A
git commit -m "chore: move experiments under notebooks"
```

---

### Task 5: Update Execution Path Contracts

**Files:**
- Modify: `scripts/verify_repo_config.yaml`
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: moved notebook paths under `notebooks/`.
- Produces: execution and static-analysis configs that point to new paths.

- [ ] **Step 1: Rewrite verifier config notebook paths**

Run:

```bash
python - <<'PY'
from pathlib import Path

path = Path("scripts/verify_repo_config.yaml")
text = path.read_text(encoding="utf-8")
active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]
for task in active:
    text = text.replace(f"  {task}/", f"  notebooks/{task}/")
    text = text.replace(f"- {task}/", f"- notebooks/{task}/")
path.write_text(text, encoding="utf-8")
PY
```

Expected: `required_sections` and `tier_a_notebooks` paths start with `notebooks/`; `active_task_dirs` remains task slugs without `notebooks/`.

- [ ] **Step 2: Rewrite Makefile notebook paths**

Run:

```bash
python - <<'PY'
from pathlib import Path

path = Path("Makefile")
text = path.read_text(encoding="utf-8")
for line in Path("scripts/verify_repo_config.yaml").read_text(encoding="utf-8").splitlines():
    pass
active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]
for task in active:
    text = text.replace(f"    {task}/", f"    notebooks/{task}/")
    text = text.replace(f"# {task}/", f"# notebooks/{task}/")
    text = text.replace(f"`{task}/", f"`notebooks/{task}/")
path.write_text(text, encoding="utf-8")
PY
```

Expected: all `TIER_A`, `TIER_B`, and `TIER_C` entries start with `notebooks/`.

- [ ] **Step 3: Rewrite CI workflow notebook paths and Docker tags**

Run:

```bash
python - <<'PY'
from pathlib import Path

path = Path(".github/workflows/ci.yml")
text = path.read_text(encoding="utf-8")
active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]
for task in active:
    text = text.replace(f"            {task}/", f"            notebooks/{task}/")
    text = text.replace(f"`{task}/", f"`notebooks/{task}/")
text = text.replace("ml-lab-ci", "ml-eng-lab-ci")
text = text.replace("ml-lab pins", "ml-eng-lab pins")
path.write_text(text, encoding="utf-8")
PY
```

Expected: artifact paths start with `notebooks/`, Docker build uses `ml-eng-lab-ci`.

- [ ] **Step 4: Update ruff per-file ignore**

Modify `pyproject.toml`:

```toml
"notebooks/node_classification-reddit-gnn-pyg/phase3-*.ipynb" = ["F401", "F541", "F811"]
```

Also update the top comment to say `ml-eng-lab`.

- [ ] **Step 5: Parse configs**

Run:

```bash
python - <<'PY'
from pathlib import Path
import tomllib
import yaml

tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
yaml.safe_load(Path("scripts/verify_repo_config.yaml").read_text(encoding="utf-8"))
yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
print("config parse ok")
PY
```

Expected:

```text
config parse ok
```

- [ ] **Step 6: Run path-contract checks**

Run:

```bash
python scripts/verify_repo.py --check docs --fast
python scripts/verify_repo.py --check execution --fast
```

Expected: no error-severity findings for missing notebooks, tier drift, or CI artifact drift.

- [ ] **Step 7: Commit**

Run:

```bash
git add scripts/verify_repo_config.yaml Makefile .github/workflows/ci.yml pyproject.toml
git commit -m "chore: update notebook execution paths"
```

---

### Task 6: Update Live Repository Identity To `ml-eng-lab`

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/jupyterhub-integration.md`
- Modify: `docs/vscode-remote-access.md`
- Modify: `docs/FINDINGS-NNX.md`
- Modify: `.devcontainer/devcontainer.json`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/start-jupyterhub.sh`
- Modify: `deploy/genai-vanilla-jupyterhub.override.yml`
- Modify: `Dockerfile` when live repository-name references are present.

**Interfaces:**
- Consumes: old live identity strings.
- Produces: live current guidance that says `ml-eng-lab`.

- [ ] **Step 1: Apply safe live identity replacements**

Run:

```bash
python - <<'PY'
from pathlib import Path

files = [
    Path("README.md"),
    Path("CONTRIBUTING.md"),
    Path("docs/env-setup.md"),
    Path("docs/jupyterhub-integration.md"),
    Path("docs/vscode-remote-access.md"),
    Path("docs/FINDINGS-NNX.md"),
    Path(".devcontainer/devcontainer.json"),
    Path(".github/workflows/ci.yml"),
    Path("scripts/start-jupyterhub.sh"),
    Path("deploy/genai-vanilla-jupyterhub.override.yml"),
    Path("Dockerfile"),
    Path("pyproject.toml"),
    Path("Makefile"),
]
replacements = {
    "github.com/thekaveh/ml-lab": "github.com/thekaveh/ml-eng-lab",
    "thekaveh/ml-lab": "thekaveh/ml-eng-lab",
    "nbviewer.org/github/thekaveh/ml-lab": "nbviewer.org/github/thekaveh/ml-eng-lab",
    "/workspaces/ml-lab": "/workspaces/ml-eng-lab",
    "/home/jovyan/work/ml-lab": "/home/jovyan/work/ml-eng-lab",
    "~/repos/ml-lab": "~/repos/ml-eng-lab",
    "ml-lab-ci": "ml-eng-lab-ci",
    "ml-lab .": "ml-eng-lab .",
    "ml-lab\n": "ml-eng-lab\n",
    "`ml-lab`": "`ml-eng-lab`",
    "# ml-lab": "# ml-eng-lab",
    " ml-lab ": " ml-eng-lab ",
    " ml-lab's ": " ml-eng-lab's ",
    "ml-lab dep set": "ml-eng-lab dep set",
    "ml-lab notebooks": "ml-eng-lab notebooks",
    "ml-lab work": "ml-eng-lab work",
    "ml-lab tree": "ml-eng-lab tree",
    "ml-lab repo": "ml-eng-lab repo",
    "ml-lab's": "ml-eng-lab's",
}
for path in files:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements.items():
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(path)
PY
```

Expected: listed files are modified. Historical entries in `CHANGELOG.md` are not touched by this step.

- [ ] **Step 2: Manually update Docker examples**

In `README.md` and `docs/env-setup.md`, ensure examples read:

```bash
docker build -t ml-eng-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

In `.github/workflows/ci.yml`, ensure:

```yaml
run: docker build -t ml-eng-lab-ci .
```

- [ ] **Step 3: Manually update devcontainer name**

In `.devcontainer/devcontainer.json`, ensure:

```json
"name": "ml-eng-lab"
```

- [ ] **Step 4: Update live clone command**

In `docs/jupyterhub-integration.md`, ensure the current clone command is:

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml-eng-lab.git
cd ml-eng-lab
scripts/start-jupyterhub.sh
```

- [ ] **Step 5: Scan live docs for old repo identity**

Run:

```bash
rg -n 'ml-lab|thekaveh/ml-lab|/workspaces/ml-lab|/home/jovyan/work/ml-lab|nbviewer.org/github/thekaveh/ml-lab' \
  README.md CONTRIBUTING.md docs/*.md .devcontainer .github scripts deploy Dockerfile Makefile pyproject.toml
```

Expected: no hits in live docs/config. If hits remain in `docs/FINDINGS-NNX.md` describing historical output leaks, keep them only when the surrounding paragraph explicitly says it is historical or recorded output.

- [ ] **Step 6: Commit**

Run:

```bash
git add README.md CONTRIBUTING.md docs .devcontainer .github scripts deploy Dockerfile Makefile pyproject.toml
git commit -m "docs: rename live repo references to ml-eng-lab"
```

---

### Task 7: Rewrite Notebook And README Links For New Paths

**Files:**
- Modify: `notebooks/**/*.md`
- Modify: `notebooks/**/*.ipynb`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/*.md`

**Interfaces:**
- Consumes: moved notebook paths.
- Produces: live links using `notebooks/` and `ml-eng-lab`.

- [ ] **Step 1: Run mechanical text path rewrite across Markdown files**

Run:

```bash
python - <<'PY'
from pathlib import Path

active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]

md_files = [Path("README.md"), Path("CONTRIBUTING.md"), Path("CHANGELOG.md")]
md_files.extend(sorted(Path("docs").glob("*.md")))
md_files.extend(sorted(Path("notebooks").glob("**/*.md")))

for path in md_files:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    for task in active:
        text = text.replace(f"https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/{task}/", f"https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/{task}/")
        text = text.replace(f"https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/{task}/", f"https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/notebooks/{task}/")
        text = text.replace(f"({task}/", f"(notebooks/{task}/")
        text = text.replace(f"`{task}/", f"`notebooks/{task}/")
    text = text.replace("(archive/README.md)", "(notebooks/archive/README.md)")
    text = text.replace("(archive/codexglue_summarization/)", "(notebooks/archive/codexglue_summarization/)")
    text = text.replace("archive/codexglue_summarization/", "notebooks/archive/codexglue_summarization/")
    if path.parts[:1] == ("notebooks",):
        text = text.replace("(../docs/", "(../../docs/")
        text = text.replace("](../docs/", "](../../docs/")
        text = text.replace("](../README.md)", "](../../README.md)")
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(path)
PY
```

Expected: Markdown links now point to `notebooks/...`.

- [ ] **Step 2: Rewrite notebook markdown cells**

Run:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]

for nb_path in sorted(Path("notebooks").glob("**/*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        source = cell.source
        original = source
        source = source.replace("https://nbviewer.org/github/thekaveh/ml-lab/", "https://nbviewer.org/github/thekaveh/ml-eng-lab/")
        source = source.replace("https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/", "https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/notebooks/")
        source = source.replace("https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/", "https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/")
        for task in active:
            source = source.replace(f"../{task}/", f"../{task}/")
            source = source.replace(f"]({task}/", f"](../{task}/")
            source = source.replace(f"`{task}/", f"`../{task}/")
        source = source.replace("../notebooks/", "../")
        if source != original:
            cell.source = source
            changed = True
    if changed:
        nbformat.write(nb, nb_path)
        print(nb_path)
PY
```

Expected: only markdown cells change. Review `git diff --stat` to confirm no code-cell source churn.

- [ ] **Step 3: Fix per-experiment README relative docs links**

Run:

```bash
rg -n '\]\(\.\./docs/|\]\(docs/|nbviewer.org/github/thekaveh/ml-lab|github.com/thekaveh/ml-lab|blob/main/[^n]' notebooks README.md CONTRIBUTING.md docs
```

Expected: no hits for old repo URLs. For remaining `../docs/` hits inside `notebooks/<task>/README.md`, replace with `../../docs/`.

- [ ] **Step 4: Run docs verifier**

Run:

```bash
python scripts/verify_repo.py --check docs --fast
```

Expected: no error-severity findings.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md CONTRIBUTING.md CHANGELOG.md docs notebooks
git commit -m "docs: update notebook links for new layout"
```

---

### Task 8: Update Runtime Documentation For New Experiment Home

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/jupyterhub-integration.md`
- Modify: `docs/vscode-remote-access.md`
- Modify: `notebooks/*/README.md`
- Modify: `notebooks/archive/README.md`

**Interfaces:**
- Consumes: new `notebooks/` layout and `ml-eng-lab` identity.
- Produces: human docs that explain the new convention.

- [ ] **Step 1: Update root README layout block**

In `README.md`, replace the repository layout block with this shape:

```text
ml-eng-lab/
├── notebooks/                                 (active and archived experiment homes)
│   ├── image_classification-mnist-ffnn-numpy/ (README, notebook, helper .py, ignored data/)
│   ├── node_classification-reddit-gnn-pyg/    (multi-phase notebooks, ignored data/)
│   └── archive/                               (preserved CodeXGLUE experiments)
├── scripts/                                   (jupyterhub start, verifier, notebook edit/import helpers)
├── tests/                                     (verifier and NNx surface tests)
├── docs/                                      (runtime, dependency, findings, maintenance docs)
├── deploy/                                    (JupyterHub compose override)
└── vendor/genai-vanilla/                      (git submodule, JupyterHub stack)
```

- [ ] **Step 2: Update contributing convention**

In `CONTRIBUTING.md`, replace the old flat-folder rule with:

```markdown
- This is a notebook-driven ML engineering lab. Each experiment lives under `notebooks/<task-dataset-model-framework>/` and is a self-contained runnable home: README, notebook(s), task-local helper code, ignored `data/`, and ignored `runs/` stay together.
- Do not add new experiment folders at the repository root. New active experiments go under `notebooks/`; preserved historical experiments go under `notebooks/archive/`.
```

- [ ] **Step 3: Update run instructions in per-experiment READMEs**

For each `notebooks/<task>/README.md`, ensure the run section says:

```markdown
Open the notebook from this experiment directory so relative `./data` and `./runs` paths resolve locally:
```

Then show the notebook path as `notebooks/<task>/<notebook>.ipynb`.

- [ ] **Step 4: Update JupyterHub docs**

In `docs/jupyterhub-integration.md`, ensure the wrapper path says:

```text
The override bind-mounts `${ML_REPO_PATH}:/home/jovyan/work/ml-eng-lab`, so from the running container's perspective, the repo is at `/home/jovyan/work/ml-eng-lab/`.
```

Also update notebook examples to:

```text
/home/jovyan/work/ml-eng-lab/notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb
```

- [ ] **Step 5: Update VS Code docs**

In `docs/vscode-remote-access.md`, ensure examples use:

```text
~/repos/ml-eng-lab/notebooks/<task>/notebook.ipynb
work/ml-eng-lab/notebooks/<task>/notebook.ipynb
```

- [ ] **Step 6: Run markdown link verifier**

Run:

```bash
python scripts/verify_repo.py --check docs --fast
```

Expected: no error-severity findings.

- [ ] **Step 7: Commit**

Run:

```bash
git add README.md CONTRIBUTING.md docs notebooks
git commit -m "docs: describe notebooks as experiment homes"
```

---

### Task 9: Update Tests For New Paths And Rename

**Files:**
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_rewrite_imports.py`
- Modify: `tests/test_edit_notebook_markdown.py`
- Modify: `tests/test_inject_smoke_test_cell.py`
- Modify: `tests/nnx_surface/*.py`

**Interfaces:**
- Consumes: moved paths and updated verifier helpers.
- Produces: test suite with no old live path assumptions.

- [ ] **Step 1: Scan tests for old paths**

Run:

```bash
rg -n '(^|["` ])(?:image_classification|node_classification|tabular_|model_surgery|quantization|pruning|knowledge_distillation|text_generation|peft|dim_reduction|diffusion|moe|clustering|link_prediction|community_detection|text_classification|sentiment_classification|preference_alignment|self_supervised)[^"` ]*/.*\.ipynb|ml-lab|thekaveh/ml-lab|archive/codexglue_summarization' tests
```

Expected: hits show concrete tests/docstrings to update.

- [ ] **Step 2: Update Makefile parser expected paths**

In `tests/test_verify_repo.py`, update synthetic Makefile expectations from:

```python
"first/notebook.ipynb"
```

to:

```python
"notebooks/first/notebook.ipynb"
```

Do the same for `second/notebook.ipynb` and artifact path tests.

- [ ] **Step 3: Update monkeypatched verifier notebook paths**

In `tests/test_verify_repo.py`, update monkeypatches such as:

```python
monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ("task/notebook.ipynb",))
```

to:

```python
monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ("notebooks/task/notebook.ipynb",))
```

- [ ] **Step 4: Update NNx surface docstrings**

Replace live docstring references to old paths with `notebooks/...`. Keep historical references only when the text explicitly describes old behavior.

- [ ] **Step 5: Run full tests**

Run:

```bash
pytest tests/ -v
```

Expected: all tests pass. Existing environment-dependent skips remain skips.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests
git commit -m "test: update path expectations for notebook layout"
```

---

### Task 10: Update Notebook Metadata And Parse All Notebooks

**Files:**
- Modify: `notebooks/**/*.ipynb`

**Interfaces:**
- Consumes: moved notebooks.
- Produces: parseable notebooks with metadata paths aligned to the new layout where metadata is live tooling output.

- [ ] **Step 1: Scan notebook metadata for old input/output paths**

Run:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

for nb_path in sorted(Path("notebooks").glob("**/*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    pm = nb.metadata.get("papermill", {})
    input_path = str(pm.get("input_path", ""))
    output_path = str(pm.get("output_path", ""))
    if "ml-lab" in input_path or "ml-lab" in output_path or (input_path and not input_path.startswith("notebooks/")):
        print(nb_path, "input=", input_path, "output=", output_path)
PY
```

Expected: a list of notebooks with stale papermill metadata.

- [ ] **Step 2: Update papermill metadata paths only**

Run:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

for nb_path in sorted(Path("notebooks").glob("**/*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    pm = nb.metadata.get("papermill")
    if not isinstance(pm, dict):
        continue
    changed = False
    new_rel = nb_path.as_posix()
    for key in ("input_path", "output_path"):
        value = pm.get(key)
        if isinstance(value, str) and (value.endswith(".ipynb") or "ml-lab" in value or "/ml/" in value):
            pm[key] = new_rel
            changed = True
    if changed:
        nbformat.write(nb, nb_path)
        print(nb_path)
PY
```

Expected: only notebook metadata changes. Do not edit cell outputs containing historical absolute paths in this task.

- [ ] **Step 3: Parse every notebook**

Run:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

count = 0
for nb_path in sorted(Path("notebooks").glob("**/*.ipynb")):
    nbformat.read(nb_path, as_version=4)
    count += 1
print(f"parsed={count}")
PY
```

Expected:

```text
parsed=51
```

- [ ] **Step 4: Run notebook structure verifier**

Run:

```bash
python scripts/verify_repo.py --check structure --fast
```

Expected: no error-severity findings.

- [ ] **Step 5: Commit**

Run:

```bash
git add notebooks
git commit -m "chore: update notebook metadata paths"
```

---

### Task 11: Stale Path And Identity Audit

**Files:**
- Modify whichever files the scans identify as live stale references.

**Interfaces:**
- Consumes: all moved and rewritten files.
- Produces: no live stale path references.

- [ ] **Step 1: Scan for old active notebook paths**

Run:

```bash
python - <<'PY'
from pathlib import Path

active = [
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
    "tabular_classification-iris-mlp-pytorch",
    "model_surgery-mnist-ffnn-pytorch",
    "quantization-mnist-ffnn-pytorch",
    "pruning-mnist-ffnn-pytorch",
    "knowledge_distillation-mnist-ffnn-pytorch",
    "text_generation-tinyshakespeare-transformer-pytorch",
    "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
    "dim_reduction-iris-autoencoder-pytorch",
    "tabular_regression-diabetes-mlp-pytorch",
    "diffusion-mnist-ddpm-pytorch",
    "moe-fmnist-mixture-of-experts-pytorch",
    "clustering-iris-kmeans-vs-ae-pytorch",
    "link_prediction-karate-graphsage-pyg",
    "community_detection-karate-louvain-vs-gnn-pyg",
    "text_classification-agnews-spacy-mlp-pytorch",
    "sentiment_classification-vader-mlp-pytorch",
    "preference_alignment-toy-dpo-pytorch",
    "self_supervised-fmnist-jepa-pytorch",
]
roots = [Path("README.md"), Path("CONTRIBUTING.md"), Path("Makefile"), Path("pyproject.toml")]
roots += sorted(Path("docs").glob("*.md"))
roots += sorted(Path("scripts").glob("*.py"))
roots += sorted(Path("tests").glob("**/*.py"))
roots += sorted(Path(".github").glob("**/*"))
roots += sorted(Path("notebooks").glob("**/*.md"))

hits = []
for path in roots:
    if not path.is_file():
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for task in active:
        old = f"{task}/"
        if old in text and f"notebooks/{task}/" not in text:
            hits.append((path, task))
            break
for path, task in hits:
    print(f"{path}: old live task path token {task}/")
if hits:
    raise SystemExit(1)
print("old active path scan ok")
PY
```

Expected:

```text
old active path scan ok
```

- [ ] **Step 2: Scan for old archive notebook paths**

Run:

```bash
rg -n 'archive/codexglue_summarization|archive/README.md' README.md CONTRIBUTING.md docs scripts tests .github notebooks --glob '!notebooks/archive/**'
```

Expected: no live hits outside text that intentionally explains the historical source path. Replace live hits with `notebooks/archive/...`.

- [ ] **Step 3: Scan for old live repo identity**

Run:

```bash
rg -n 'github.com/thekaveh/ml-lab|nbviewer.org/github/thekaveh/ml-lab|/workspaces/ml-lab|/home/jovyan/work/ml-lab|ml-lab-ci|docker build -t ml-lab|docker run .* ml-lab|# ml-lab|`ml-lab`' \
  README.md CONTRIBUTING.md docs scripts tests .github .devcontainer deploy Dockerfile Makefile pyproject.toml notebooks
```

Expected: no output. If output appears in `docs/FINDINGS-NNX.md` as historical leakage discussion, reword the paragraph so it says the old string is historical, or move it to a historical exception list in the final report.

- [ ] **Step 4: Add changelog entry**

At the top of `CHANGELOG.md` under `[Unreleased]`, add:

```markdown
- **Repository layout and identity migration**: active and archived experiment notebooks now live under `notebooks/`, with each experiment directory acting as the runnable home for its README, notebook(s), helper code, and ignored local artifacts. Live repository identity references now target `ml-eng-lab` rather than `ml-lab`, including GitHub URLs, nbviewer links, Docker tags, Codespaces paths, and JupyterHub bind-mount examples.
```

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md CONTRIBUTING.md CHANGELOG.md docs scripts tests .github .devcontainer deploy Dockerfile Makefile pyproject.toml notebooks
git commit -m "chore: remove stale notebook layout references"
```

---

### Task 12: Full Verification And Final Fixes

**Files:**
- Modify only files required by failing verification.

**Interfaces:**
- Consumes: migrated repo.
- Produces: green verification or explicit environment-limited notes.

- [ ] **Step 1: Run config and shell syntax checks**

Run:

```bash
python - <<'PY'
from pathlib import Path
import tomllib
import yaml

tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
yaml.safe_load(Path("scripts/verify_repo_config.yaml").read_text(encoding="utf-8"))
yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
yaml.safe_load(Path("deploy/genai-vanilla-jupyterhub.override.yml").read_text(encoding="utf-8"))
print("config parse ok")
PY
bash -n scripts/start-jupyterhub.sh
git diff --check
```

Expected:

```text
config parse ok
```

`bash -n` and `git diff --check` exit 0.

- [ ] **Step 2: Run verifier**

Run:

```bash
python scripts/verify_repo.py --check all --fast
```

Expected: exit 0 with no error-severity findings.

- [ ] **Step 3: Run tests**

Run:

```bash
pytest tests/ -v
```

Expected: all tests pass. Existing known skips remain skips.

- [ ] **Step 4: Run lint**

Run:

```bash
ruff check . --no-cache
```

Expected: exit 0.

- [ ] **Step 5: Check Tier-A clean target**

Run:

```bash
make check-tier-a-clean
```

Expected: exit 0. This target should check the new `notebooks/...` Tier-A paths.

- [ ] **Step 6: Run focused papermill smoke for the sibling-import notebook**

Run:

```bash
mkdir -p /tmp/ml-eng-lab-smoke
(
  cd notebooks/image_classification-mnist-ffnn-numpy
  papermill --kernel python3 -p SMOKE_TEST 1 notebook.ipynb /tmp/ml-eng-lab-smoke/numpy-smoke.ipynb
)
```

Expected: papermill completes. This validates that sibling `.py` imports and local `./data` resolution work from the new directory. If the local environment lacks the heavyweight notebook runtime, record the exact import error and continue with the static verification already run.

- [ ] **Step 7: Run final stale scans**

Run:

```bash
git ls-files '*.ipynb' | awk '$0 !~ /^notebooks\// {print}'
rg -n 'github.com/thekaveh/ml-lab|nbviewer.org/github/thekaveh/ml-lab|/workspaces/ml-lab|/home/jovyan/work/ml-lab|ml-lab-ci|docker build -t ml-lab|docker run .* ml-lab' \
  README.md CONTRIBUTING.md docs scripts tests .github .devcontainer deploy Dockerfile Makefile pyproject.toml notebooks
```

Expected: both commands produce no output except documented historical exceptions. Historical exceptions must be listed in the final response.

- [ ] **Step 8: Commit final fixes**

If verification required fixes, run:

```bash
git add -A
git commit -m "fix: complete notebook layout verification"
```

If no files changed, do not commit.

- [ ] **Step 9: Final status**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: clean working tree on `codex/overnight-maintenance`.

---

## Self-Review

Spec coverage:

- `notebooks/` active layout: Tasks 4, 5, 7, 8, 11, 12.
- Archived CodeXGLUE move: Tasks 4, 7, 11, 12.
- Per-experiment README/helper/runtime artifacts: Tasks 4, 8, 12.
- Repo rename to `ml-eng-lab`: Tasks 6, 7, 8, 11, 12.
- Tooling and CI: Tasks 2, 3, 5, 9, 12.
- Verification: Task 12.
- Historical exceptions policy: Tasks 6, 11, 12.

Placeholder scan:

- The plan contains no unresolved placeholder tokens and no open-ended repair steps.
- Every code-editing task includes concrete snippets or exact rewrite commands.

Type and interface consistency:

- `NOTEBOOK_ROOT`, `ARCHIVE_NOTEBOOK_ROOT`, `_active_task_path`, and `_baseline_notebook_rel` are introduced in Task 2 and used by later verifier updates.
- `active_task_dirs` remains task slugs, while notebook path lists use `notebooks/<task>/...`.
- Archive exclusion consistently uses `notebooks/archive`.
