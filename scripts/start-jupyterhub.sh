#!/usr/bin/env bash
# Start genai-vanilla (vendored at vendor/genai-vanilla, pinned to main)
# with the ml-integration override layered on top.
#
# Usage:
#   ml/scripts/start-jupyterhub.sh           # equivalent to running ./start.sh in the submodule
#   ml/scripts/start-jupyterhub.sh <args>    # extra args passed through
set -euo pipefail

ML_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENAI_DIR="$ML_REPO_ROOT/vendor/genai-vanilla"
OVERRIDE_FILE="$ML_REPO_ROOT/deploy/genai-vanilla-jupyterhub.override.yml"

if [ ! -d "$GENAI_DIR" ]; then
    echo "ERROR: $GENAI_DIR not found." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

if [ ! -f "$OVERRIDE_FILE" ]; then
    echo "ERROR: $OVERRIDE_FILE not found." >&2
    exit 1
fi

# Required by the override file:
export ML_REPO_PATH="$ML_REPO_ROOT"
export HOST_SSH_DIR="${HOST_SSH_DIR:-$HOME/.ssh}"

# Tell docker compose to layer our override on top of the base compose.
# COMPOSE_FILE is a colon-separated list (Unix); absolute paths are fine.
export COMPOSE_FILE="$GENAI_DIR/docker-compose.yml:$OVERRIDE_FILE"

# IMPORTANT: cd into the genai-vanilla directory so its start.sh works
# from its expected CWD (loads its own .env, finds its services/ tree, etc.).
cd "$GENAI_DIR"
exec ./start.sh "$@"
