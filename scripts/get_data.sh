#!/usr/bin/env bash
# Alert history: Vadimkin/ukrainian-air-raid-sirens-dataset, pinned to the snapshot used for the reported
# results (downloaded 2026-10-11). Pass "latest" to use today's data instead.
set -euo pipefail
cd "$(dirname "$0")/.."
REV="${1:-bf0c323c78e3e4508721458797dcd38c95df13f3}"
if [ ! -d data/sirens/.git ]; then
  git clone https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git data/sirens
fi
if [ "$REV" = "latest" ]; then
  git -C data/sirens pull --ff-only
else
  git -C data/sirens fetch --quiet origin "$REV" 2>/dev/null || git -C data/sirens fetch --quiet --unshallow || true
  git -C data/sirens checkout --quiet "$REV"
fi
echo "alert data at $(git -C data/sirens rev-parse --short HEAD)"
