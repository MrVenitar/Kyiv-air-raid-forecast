#!/usr/bin/env bash
# Alert history: Vadimkin/ukrainian-air-raid-sirens-dataset, pinned to the snapshot used for the Kyiv results
# (downloaded 2026-10-12). The 24-location reports in reports/volunteer|official were produced with the
# 2026-10-11 snapshot: bash scripts/get_data.sh bf0c323c78e3e4508721458797dcd38c95df13f3
# Pass "latest" to use today's data instead.
set -euo pipefail
cd "$(dirname "$0")/.."
REV="${1:-d10c80fa696bab42822b9f05b423fc8a30e79d20}"
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
