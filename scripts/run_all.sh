#!/usr/bin/env bash
# Part 2 (24 places, hourly): data snapshot of 2026-10-11, leakage tests, both pipelines, figures + summary
# (about 15 min on 2 CPUs). The Kyiv forecaster has its own steps (README, Part 1 / kyiv/colab_train.ipynb).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/get_data.sh bf0c323c78e3e4508721458797dcd38c95df13f3
python3 tests/test_leakage.py
OMP_NUM_THREADS=1 python3 src/run.py --source volunteer
OMP_NUM_THREADS=1 python3 src/run.py --source official
python3 src/report.py
