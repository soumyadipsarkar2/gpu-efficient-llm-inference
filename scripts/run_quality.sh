#!/usr/bin/env bash
# E2: model quality per precision. fp32 (0.5B) / fp16 (1.5B) are the agreement references.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
O=results/quality
S=Qwen/Qwen2.5-0.5B-Instruct
L=Qwen/Qwen2.5-1.5B-Instruct
run() { [ -f "$O/$1.json" ] || $PY -m bench.quality "${@:2}" --out "$O/$1.json"; }
run qwen0.5b_fp32 --model $S --precision fp32
for p in fp16 bf16 int8 int4; do run qwen0.5b_$p --model $S --precision $p --ref $O/qwen0.5b_fp32.json; done
run qwen0.5b_int4_noemb --model $S --precision int4 --no-quantize-embeddings --ref $O/qwen0.5b_fp32.json
run qwen1.5b_fp16 --model $L --precision fp16 --arc-limit 1000
for p in int8 int4; do run qwen1.5b_$p --model $L --precision $p --arc-limit 1000 --ref $O/qwen1.5b_fp16.json; done
