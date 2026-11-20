#!/usr/bin/env bash
# Resume after interruption: clean 0.5B profiles first, then the rest of the pipeline.
cd "$(dirname "$0")/.."
exec caffeinate -i scripts/run_all.sh
