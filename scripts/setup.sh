#!/usr/bin/env bash
# Install the pre-commit hook.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

hook_src="${HERE}/pre-commit"
hook_dest="$(git rev-parse --git-dir)/hooks/pre-commit"
chmod +x "${hook_src}"
ln -sf "${hook_src}" "${hook_dest}"
