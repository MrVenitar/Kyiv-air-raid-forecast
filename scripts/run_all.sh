#!/usr/bin/env bash
# Full reproduction: data, leakage tests, both pipelines, figures + summary (about 15 min on 2 CPUs).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/get_data.sh
python3 tests/test_leakage.py
OMP_NUM_THREADS=1 python3 src/run.py --source volunteer
OMP_NUM_THREADS=1 python3 src/run.py --source official
python3 src/report.py
