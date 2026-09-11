#!/bin/bash
# ToolVoiceMac v0.2.0. Installs an isolated tool; never writes user jobs or voices.
set -euo pipefail
version='0.2.0'
repo='https://github.com/nthieuu205z/ToolVoiceMac'
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'ToolVoiceMac yêu cầu Mac Apple Silicon.' >&2
  exit 1
fi
mac_major="$(sw_vers -productVersion)"
if (( ${mac_major%%.*} < 14 )); then
  printf '%s\n' 'ToolVoiceMac yêu cầu macOS 14 trở lên.' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    printf '%s\n' 'Hãy cài Homebrew từ https://brew.sh rồi chạy: brew install ffmpeg' >&2
    exit 1
  fi
  brew install ffmpeg
fi
if ! command -v uv >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
task_tmp="$(mktemp -d)"
trap 'rm -rf -- "$task_tmp"' EXIT
curl --proto '=https' --tlsv1.2 -fLsS "https://raw.githubusercontent.com/nthieuu205z/ToolVoiceMac/v${version}/requirements-macos.lock" -o "$task_tmp/requirements-macos.lock"
uv tool install --python 3.12.14 --with-requirements "$task_tmp/requirements-macos.lock" "${repo}/archive/refs/tags/v${version}.tar.gz"
uv tool update-shell
printf '\n%s\n' 'Đã cài ToolVoiceMac. Mở Terminal mới và gõ: toolvoice'
