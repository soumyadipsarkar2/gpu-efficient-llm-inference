#!/usr/bin/env bash
# Re-measure the 0.5B profiles on an idle GPU (the first pass overlapped with other jobs),
# then refresh the calibrated simulator results and figures.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p results/profile/first_pass
mv results/profile/qwen0.5b_*.json results/profile/first_pass/ 2>/dev/null || true
scripts/run_profiles.sh
rm -rf results/sim
$PY -m bench.sim_experiments
$PY -m analysis.figures
