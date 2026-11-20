#!/usr/bin/env bash
# Reproduce every result in the report (about 4-5 hours on an Apple M4; finished steps are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
scripts/run_profiles.sh                    # E0, E1 (+ simulator calibration)
scripts/run_quality.sh                     # E2
$PY -m bench.experiments batching          # E3
$PY -m bench.experiments scaling           # E4
$PY -m bench.experiments routing           # E5
$PY -m bench.experiments fault             # E6
$PY -m bench.sim_experiments               # E7
$PY -m analysis.figures                    # figures + tables
