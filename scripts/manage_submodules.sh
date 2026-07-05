#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cmd="${1:-status}"

usage() {
  cat <<'EOF'
Usage: bash scripts/manage_submodules.sh <command>

Commands:
  status       Show recursive submodule status.
  init         Sync URLs and initialize all submodules at recorded commits.
  update       Update initialized submodules from their configured branch.

Notes:
  - Parent commits record exact submodule commits for reproducibility.
  - Run "update" only when you intentionally want newer submodule commits.
  - Commit submodule changes in the submodule repo first, then commit the
    updated gitlink in this parent repo.
EOF
}

case "$cmd" in
  status)
    git -C "$REPO_ROOT" submodule status --recursive
    ;;
  init)
    git -C "$REPO_ROOT" submodule sync --recursive
    git -C "$REPO_ROOT" submodule update --init --recursive
    ;;
  update)
    git -C "$REPO_ROOT" submodule sync --recursive
    git -C "$REPO_ROOT" submodule update --init --recursive --remote
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
