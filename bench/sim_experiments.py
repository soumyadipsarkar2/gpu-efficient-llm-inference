"""Simulator experiments (E7): routing on larger clusters of independent GPUs.

Uses step-time/prefill models fitted from the real profiles in
``results/profile`` and the same routing-policy code as the live router.

    S1 validate  - simulate the single-worker open-loop runs of E3 and compare with measurements
    S2 load      - 8-worker heterogeneous cluster, offered-load sweep, 4 policies x 3 workloads x 5 seeds
    S3 homo      - same sweep on a homogeneous 8-worker cluster (GPU-aware should not hurt)
    S4 stale     - sensitivity to telemetry staleness (poll interval), incl. a router that only
                   sees worker-reported queue lengths (no router-local counters)

    python -m bench.sim_experiments
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llmserve.sim.simulator import Profile, SimReq, simulate, synthetic

ROOT = Path(__file__).resolve().parents[1]
PROF = ROOT / "results" / "profile"
OUT = ROOT / "results" / "sim"
POLICIES = ["random", "round_robin", "least_outstanding", "gpu_aware"]
SEEDS = range(5)


def prof(name: str, **kw) -> Profile:
    return Profile.from_json(PROF / f"{name}.json", name=kw.pop("label", name), **kw)


def hetero_cluster() -> list[Profile]:
    """8 'GPUs': fast fp16 workers, int8 workers (slower, half the weight memory),
    memory-starved fp16 workers, and an older/slower GPU class (2x step time)."""
    return (
        [prof("qwen0.5b_fp16", label=f"fp16-{i}", max_batch=32, kv_budget_tokens=65536) for i in range(3)]
        + [prof("qwen0.5b_int8", label=f"int8-{i}", max_batch=32, kv_budget_tokens=65536) for i in range(2)]
        + [prof("qwen0.5b_fp16", label=f"fp16-small-{i}", max_batch=8, kv_budget_tokens=8192) for i in range(2)]
        + [prof("qwen0.5b_fp16", label="fp16-oldgpu", max_batch=32, kv_budget_tokens=65536, speed=2.0)]
    )


def homo_cluster() -> list[Profile]:
    return [prof("qwen0.5b_fp16", label=f"fp16-{i}", max_batch=32, kv_budget_tokens=65536) for i in range(8)]


def s1_validate() -> dict:
    """Compare simulated vs measured latency for the single-worker open-loop runs."""
    from bench.workload import load

    out = {}
    p = prof("qwen0.5b_fp16", max_batch=32, kv_budget_tokens=65536)
    for rate in [1.0, 2.0, 4.0]:
        real_f = ROOT / "results" / "batching" / "open" / f"continuous_b32_r{rate:g}.json"
        tr_f = ROOT / "results" / "traces" / f"short_poisson_r{rate:g}.jsonl"
        if not real_f.exists() or not tr_f.exists():
            continue
        tr = load(tr_f)
        reqs = [SimReq(r.idx, r.t, r.prompt_tokens, r.max_tokens, r.cls) for r in tr]
        sim = simulate([p], reqs, "round_robin").summary()
        real = json.loads(real_f.read_text())["summary"]
        out[f"r{rate:g}"] = {k: {"sim": sim[k], "real": real[k]} for k in
                             ["e2e_p50", "e2e_p95", "ttft_p50", "ttft_p95", "tpot_p50", "throughput_tok_s"]}
    return out


def sweep(cluster: list[Profile], rates, workloads=("poisson", "bursty", "heavy_tail"), n: int = 3000) -> list[dict]:
    rows = []
    for wl in workloads:
        mix = "heavy_tail" if wl == "heavy_tail" else "balanced"
        arr = "bursty" if wl == "bursty" else "poisson"
        for rate in rates:
            for seed in SEEDS:
                reqs = synthetic(n, mix, arr, rate=rate, seed=seed, burst=6.0)
                for pol in POLICIES:
                    s = simulate(cluster, reqs, pol, poll_interval=0.1, seed=seed).summary()
                    rows.append({"workload": wl, "rate": rate, "seed": seed, "policy": pol,
                                 **{k: s[k] for k in s if k not in ("busy", "routed")},
                                 "routed": s["routed"], "busy": s["busy"]})
        print(f"  {wl} done", flush=True)
    return rows


def s4_stale(cluster: list[Profile], rate: float, n: int = 3000) -> list[dict]:
    rows = []
    for poll in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        for seed in SEEDS:
            reqs = synthetic(n, "balanced", "bursty", rate=rate, seed=seed, burst=6.0)
            for pol, stale in [("gpu_aware", False), ("least_outstanding", True), ("gpu_aware", True)]:
                s = simulate(cluster, reqs, pol, poll_interval=poll, seed=seed, stale_inflight=stale).summary()
                rows.append({"poll": poll, "seed": seed, "policy": pol + ("_polled" if stale else ""),
                             "e2e_p95": s["e2e_p95"], "e2e_p50": s["e2e_p50"], "ttft_p95": s["ttft_p95"]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[S1] validation", flush=True)
    (OUT / "s1_validate.json").write_text(json.dumps(s1_validate(), indent=1))
    het = hetero_cluster()
    print("[S2] hetero load sweep", flush=True)
    (OUT / "s2_hetero.json").write_text(json.dumps(sweep(het, [4, 8, 12, 16, 20, 24])))
    print("[S3] homo load sweep", flush=True)
    (OUT / "s3_homo.json").write_text(json.dumps(sweep(homo_cluster(), [8, 16, 24, 32], workloads=("poisson", "bursty"))))
    print("[S4] staleness", flush=True)
    (OUT / "s4_stale.json").write_text(json.dumps(s4_stale(het, 16)))
    _ = np  # keep numpy import for interactive use


if __name__ == "__main__":
    main()
