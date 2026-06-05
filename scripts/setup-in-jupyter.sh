#!/usr/bin/env bash
# Developer-editable-install override for nnx inside the jupyterhub container.
#
# **You probably do NOT need to run this.** As of genai-vanilla commit
# cbad341 (PR #26, 2026-06-02), the jupyterhub image ships `nnx-pytorch`
# (imported as `nnx`) baked in via pip. Notebooks just work — see
# docs/jupyterhub-integration.md §1.
#
# This script is only useful in a narrow case: you're actively editing the
# nnx source under ml-lab's `nnx/` submodule and want your edits to land in
# the running kernel WITHOUT a pip release + image rebuild cycle. It replaces
# the image's pip-installed `nnx-pytorch` with an editable install pointing
# at the bind-mounted submodule. The override persists for the lifetime of
# the container (next image rebuild wipes it).
#
# Usage (from the host):
#   docker exec -it <project>-jupyterhub /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
#
# Or (from inside the container terminal, e.g. attached VS Code):
#   /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
#
# Requires the §2 wrapper-and-bind-mount path of docs/jupyterhub-integration.md
# (the standalone-genai-vanilla §1 path has no bind-mount, so this script has
# nothing to install).
set -euo pipefail

REPO_ROOT="/home/jovyan/work/ml-lab"

if [ ! -d "$REPO_ROOT/nnx" ]; then
    echo "ERROR: $REPO_ROOT/nnx not found." >&2
    echo >&2
    echo "This script is the nnx editable-install developer override and requires" >&2
    echo "the ml-lab repo + nnx submodule to be bind-mounted into the container." >&2
    echo >&2
    echo "If you just want notebooks to work, you do NOT need this script —" >&2
    echo "the genai-vanilla image pip-installs nnx natively as of PR #26." >&2
    echo "See docs/jupyterhub-integration.md §1 for the standalone path." >&2
    echo >&2
    echo "If you want the editable-install workflow, ensure:" >&2
    echo "  - You started the stack via ml-lab's scripts/start-jupyterhub.sh" >&2
    echo "    (which sets ML_REPO_PATH and layers the bind-mount override)." >&2
    echo "  - The nnx submodule is initialized on the host:" >&2
    echo "    cd <ml-lab repo> && git submodule update --init --recursive" >&2
    exit 1
fi

echo "Overriding pip-installed nnx with editable install from $REPO_ROOT/nnx ..."
# `[lm]` extra mirrors requirements.txt and keeps the text_generation Tier-A
# notebook's `tokenizers`/`datasets` deps installed (issue #12). Without it,
# the override silently strips those packages and re-trips NNTokenizerParams.
pip install -e "$REPO_ROOT/nnx[lm]"

echo
echo "Verifying import..."
python -c "import nnx; print(f'nnx imported from {nnx.__file__}')"

echo
echo "Editable install active. Your edits under nnx/src/nnx/ are now live in the"
echo "running kernel. This override persists for the lifetime of the container;"
echo "the next image rebuild restores the pip-installed nnx-pytorch."
