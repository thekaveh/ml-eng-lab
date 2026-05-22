#!/usr/bin/env bash
# Symlinks deploy/genai-vanilla-jupyterhub.override.yml into a local
# genai-vanilla checkout as docker-compose.override.yml.
#
# Usage:
#   GENAI_VANILLA_DIR=/Users/kaveh/repos/genai-vanilla ./scripts/link-jupyter-override.sh
#
# If GENAI_VANILLA_DIR is unset, defaults to ../genai-vanilla relative
# to this repo's root.
set -euo pipefail

ML_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY_SOURCE="$ML_REPO_ROOT/deploy/genai-vanilla-jupyterhub.override.yml"

GENAI_DIR="${GENAI_VANILLA_DIR:-$(cd "$ML_REPO_ROOT/../genai-vanilla" 2>/dev/null && pwd || true)}"

if [ -z "$GENAI_DIR" ] || [ ! -d "$GENAI_DIR" ]; then
    echo "ERROR: genai-vanilla checkout not found." >&2
    echo "Set GENAI_VANILLA_DIR to its absolute path and re-run." >&2
    exit 1
fi

if [ ! -f "$OVERLAY_SOURCE" ]; then
    echo "ERROR: overlay source not found at $OVERLAY_SOURCE" >&2
    exit 1
fi

TARGET="$GENAI_DIR/docker-compose.override.yml"

if [ -e "$TARGET" ] && [ ! -L "$TARGET" ]; then
    echo "ERROR: $TARGET already exists and is not a symlink." >&2
    echo "Move or remove it before linking." >&2
    exit 1
fi

ln -sf "$OVERLAY_SOURCE" "$TARGET"

echo "Linked:  $TARGET"
echo "      → $OVERLAY_SOURCE"
echo
echo "Next:"
echo "  1) Ensure these are set in $GENAI_DIR/.env:"
echo "       ML_REPO_PATH=$ML_REPO_ROOT"
echo "       HOST_SSH_DIR=\$HOME/.ssh"
echo "  2) cd $GENAI_DIR && ./start.sh"
