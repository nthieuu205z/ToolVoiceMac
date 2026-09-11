#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
if command -v toolvoice >/dev/null 2>&1; then
  exec toolvoice "$@"
fi
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  cd "$project_dir"
  exec "$project_dir/.venv/bin/python" -m toolvoice "$@"
fi
printf '%s\n' 'Chưa cài ToolVoiceMac. Xem README.md để cài bằng Terminal.' >&2
exit 1
