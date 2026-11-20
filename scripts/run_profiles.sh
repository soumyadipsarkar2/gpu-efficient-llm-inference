#!/usr/bin/env bash
# E0/E1: profile every model/precision configuration (also calibrates the simulator).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/profile
S=Qwen/Qwen2.5-0.5B-Instruct
L=Qwen/Qwen2.5-1.5B-Instruct
run() { [ -f "$OUT/$1.json" ] || $PY -m bench.profile_engine "${@:2}" --out "$OUT/$1.json"; }
for p in fp16 bf16 fp32 int8 int4; do run qwen0.5b_$p --model $S --precision $p; done
run qwen0.5b_int4_noemb --model $S --precision int4 --no-quantize-embeddings
run qwen0.5b_fp16_hfgqa --model $S --precision fp16 --no-fast-gqa
for p in fp16 int8 int4; do run qwen1.5b_$p --model $L --precision $p --batches 1,2,4,8,16,32; done
