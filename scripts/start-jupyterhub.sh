#!/usr/bin/env bash
# Start genai-vanilla (vendored at vendor/genai-vanilla, pinned to main)
# with the ml-eng-lab integration override layered on top.
#
# Usage:
#   scripts/start-jupyterhub.sh           # no host SSH keys mounted by default
#   HOST_SSH_DIR="$HOME/.ssh" scripts/start-jupyterhub.sh  # opt-in SSH mount
#   scripts/start-jupyterhub.sh <args>    # extra args passed through
set -euo pipefail

ML_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENAI_DIR="$ML_REPO_ROOT/vendor/genai-vanilla"
OVERRIDE_FILE="$ML_REPO_ROOT/deploy/genai-vanilla-jupyterhub.override.yml"

if [ ! -f "$GENAI_DIR/start.sh" ] || [ ! -f "$GENAI_DIR/docker-compose.yml" ]; then
    echo "ERROR: genai-vanilla submodule is not initialized at $GENAI_DIR." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

if [ ! -f "$OVERRIDE_FILE" ]; then
    echo "ERROR: $OVERRIDE_FILE not found." >&2
    exit 1
fi

# Required by the override file.
export ML_REPO_PATH="$ML_REPO_ROOT"
if [ -n "${HOST_SSH_DIR:-}" ]; then
    if [ ! -d "$HOST_SSH_DIR" ]; then
        echo "ERROR: HOST_SSH_DIR does not exist: $HOST_SSH_DIR" >&2
        exit 1
    fi
    export ML_SSH_MOUNT_DIR="$HOST_SSH_DIR"
else
    export ML_SSH_MOUNT_DIR="$ML_REPO_ROOT/.claude/empty-ssh"
    mkdir -p "$ML_SSH_MOUNT_DIR"
fi

# Tell docker compose to layer our override on top of the base compose.
# COMPOSE_FILE is a colon-separated list (Unix); absolute paths are fine.
export COMPOSE_FILE="$GENAI_DIR/docker-compose.yml:$OVERRIDE_FILE"

# IMPORTANT: cd into the genai-vanilla directory so its start.sh works
# from its expected CWD (loads its own .env, finds its services/ tree, etc.).
cd "$GENAI_DIR"
exec ./start.sh "$@"
