"""Micro-benchmark one model configuration (Experiments E0/E1, simulator calibration).

Measures, for a given model + precision:
  * load time and resident weight bytes
  * prefill latency for prompt lengths L
  * decode step latency on a grid of batch size B x cached context T
    (one forward pass for B sequences + LM head + argmax, as the engine does)

and fits the step-time model used by the GPU-aware router and the simulator::

    t_step(B, T) = a + c*B + e*B*T          prefill(L) = p0 + p1*L

Usage:
    python -m bench.profile_engine --precision fp16 --out results/profile/qwen0.5b_fp16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from llmserve import device as dev
from llmserve.model_loader import load_model


def _decode_time(lm, B: int, T: int, steps: int = 8) -> float:
    base, head, d = lm.model.model, lm.model.lm_head, lm.device
    ids = torch.randint(1000, 20000, (1, T), device=d)
    cache = DynamicCache()
    base(input_ids=ids, past_key_values=cache, use_cache=True)
    for li in range(len(cache.key_cache)):  # replicate one prefilled row to B rows
        cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
        cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
    mask = torch.ones((B, T), dtype=torch.long, device=d)
    tok = torch.full((B, 1), 42, device=d)
    times = []
    for s in range(steps + 2):
        mask = torch.cat((mask, torch.ones((B, 1), dtype=torch.long, device=d)), 1)
        pos = torch.full((B, 1), T + s, device=d)
        dev.synchronize(d)
        t0 = time.perf_counter()
        out = base(input_ids=tok, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
        head(out.last_hidden_state[:, -1]).float().argmax(-1).tolist()
        dev.synchronize(d)
        times.append(time.perf_counter() - t0)
    del cache
    dev.empty_cache(d)
    return statistics.median(times[2:])


def _prefill_time(lm, L: int, reps: int = 3) -> float:
    base, head, d = lm.model.model, lm.model.lm_head, lm.device
    times = []
    for _ in range(reps + 1):
        ids = torch.randint(1000, 20000, (1, L), device=d)
        dev.synchronize(d)
        t0 = time.perf_counter()
        out = base(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        head(out.last_hidden_state[:, -1]).float().argmax(-1).tolist()
        dev.synchronize(d)
        times.append(time.perf_counter() - t0)
    dev.empty_cache(d)
    return statistics.median(times[1:])


def fit(decode: list[dict], prefill: list[dict]) -> dict:
    X = np.array([[1.0, r["B"], r["B"] * r["T"]] for r in decode])
    y = np.array([r["s"] for r in decode])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Xp = np.array([[1.0, r["L"]] for r in prefill])
    yp = np.array([r["s"] for r in prefill])
    cp, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
    return {"a": float(coef[0]), "c": float(coef[1]), "e": float(coef[2]), "r2": float(r2),
            "p0": float(cp[0]), "p1": float(cp[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--no-fast-gqa", action="store_true")
    ap.add_argument("--no-quantize-embeddings", action="store_true")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--contexts", default="128,512,1024,2048")
    ap.add_argument("--prompts", default="32,128,256,512,1024,2048")
    ap.add_argument("--max-batch-tokens", type=int, default=65536,
                    help="skip grid points with B*T above this (MPS pages beyond ~2 GB of KV)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    lm = load_model(a.model, a.precision, fast_gqa=not a.no_fast_gqa,
                    quantize_embeddings=not a.no_quantize_embeddings)
    print(json.dumps(lm.info), flush=True)
    prefill = []
    for L in map(int, a.prompts.split(",")):
        s = _prefill_time(lm, L)
        prefill.append({"L": L, "s": s})
        print(f"prefill L={L}: {s*1000:.1f} ms", flush=True)
    decode = []
    for T in map(int, a.contexts.split(",")):
        for B in map(int, a.batches.split(",")):
            if B * T > a.max_batch_tokens:
                continue
            s = _decode_time(lm, B, T)
            decode.append({"B": B, "T": T, "s": s})
            print(f"decode B={B} T={T}: {s*1000:.1f} ms  ({B/s:.0f} tok/s)", flush=True)
    res = {"info": lm.info, "fast_gqa": not a.no_fast_gqa, "prefill": prefill, "decode": decode,
           "fit": fit(decode, prefill)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("fit:", res["fit"])


if __name__ == "__main__":
    main()
