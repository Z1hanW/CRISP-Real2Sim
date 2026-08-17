#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/prep/MogeSAM"
PATCH_FILE="$REPO_ROOT/patches/mogesam-ufm-optional-vis.patch"
EXPECTED_COMMIT="6d734c85a2d0b521d8085ce87fcf5c743fee0a77"

if ! git -C "$SUBMODULE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "MogeSAM is not initialized; run: git submodule update --init prep/MogeSAM" >&2
    exit 1
fi

actual_commit="$(git -C "$SUBMODULE_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "MogeSAM is at $actual_commit; expected $EXPECTED_COMMIT." >&2
    exit 1
fi

if git -C "$SUBMODULE_DIR" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "MogeSAM agentic patch is already applied."
    exit 0
fi

if ! git -C "$SUBMODULE_DIR" apply --unidiff-zero --check "$PATCH_FILE"; then
    echo "MogeSAM agentic patch no longer applies cleanly; check the submodule revision." >&2
    exit 1
fi

git -C "$SUBMODULE_DIR" apply --unidiff-zero "$PATCH_FILE"
echo "Applied MogeSAM agentic patch."
