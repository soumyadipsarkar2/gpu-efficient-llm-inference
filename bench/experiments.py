"""Live end-to-end experiments (E3 batching, E4 scaling, E5 routing, E6 fault tolerance).

Each experiment launches real worker/router processes via :class:`llmserve.cluster.Cluster`,
replays fixed traces with the async client and writes one JSON per run to
``results/<experiment>/``. Re-running skips runs whose output already exists.

    python -m bench.experiments batching
    python -m bench.experiments scaling
    python -m bench.experiments routing
    python -m bench.experiments fault
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from llmserve.cluster import Cluster, WorkerSpec

from .client import run_trace, save_results, summarize
from .workload import generate, load, save

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TRACES = RES / "traces"


def trace(name: str, **kw):
    p = TRACES / f"{name}.jsonl"
    if not p.exists():
        save(generate(**kw), p)
    return load(p)


def _run(url, tr, out: Path, meta: dict, mode: str, concurrency: int = 1, worker_api=False, stats_urls=(),
         time_scale: float = 1.0):
    t0 = time.time()
    results, samples, makespan = asyncio.run(run_trace(url, tr, mode=mode, concurrency=concurrency,
                                                       worker_api=worker_api, stats_urls=list(stats_urls),
                                                       time_scale=time_scale))
    s = summarize(results, makespan, samples)
    save_results(out, results, samples, s, meta)
    print(f"  {out.name}: ok={s['n_ok']}/{s['n']} thr={s.get('throughput_tok_s', 0):.0f} tok/s "
          f"e2e_p50={s.get('e2e_p50', 0):.2f}s p95={s.get('e2e_p95', 0):.2f}s "
          f"ttft_p50={s.get('ttft_p50', 0):.2f}s p95={s.get('ttft_p95', 0):.2f}s  [{time.time()-t0:.0f}s]",
          flush=True)
    return s


# --------------------------------------------------------------------------- #
# E3 - batching policies on a single worker
# --------------------------------------------------------------------------- #
BATCH_CONFIGS = [
    ("serial", dict(policy="serial", max_batch=1), [1, 2, 4, 8]),
    ("static_b8_w50", dict(policy="static", max_batch=8, window_ms=50), [1, 4, 8, 16, 32]),
    ("static_b32_w50", dict(policy="static", max_batch=32, window_ms=50), [1, 4, 8, 16, 32, 64]),
    ("continuous_b8", dict(policy="continuous", max_batch=8), [1, 4, 8, 16, 32]),
    ("continuous_b32", dict(policy="continuous", max_batch=32), [1, 4, 8, 16, 32, 64]),
    ("continuous_b64", dict(policy="continuous", max_batch=64), [1, 4, 8, 16, 32, 64]),
]


def exp_batching(port: int = 8201) -> None:
    out_dir = RES / "batching"
    base = trace("balanced_closed_256", n=256, mix="balanced", arrival="closed", seed=11)
    for name, cfg, levels in BATCH_CONFIGS:
        todo = [c for c in levels if not (out_dir / f"{name}_c{c}.json").exists()]
        if not todo:
            continue
        print(f"[batching] {name}", flush=True)
        spec = WorkerSpec(port=port, **cfg)
        with Cluster([spec], router_policy=None) as cl:
            for c in todo:
                n = min(max(4 * c, 24), 256)
                _run(spec.url, base[:n], out_dir / f"{name}_c{c}.json",
                     {"exp": "batching", "config": name, "concurrency": c, **cfg}, "closed", c,
                     worker_api=True, stats_urls=[spec.url + "/stats"])

    # open-loop: batching window sweep at a fixed arrival rate (TTFT vs window)
    rates = [1.0, 2.0, 4.0]
    for rate in rates:
        tr = trace(f"short_poisson_r{rate:g}", n=120, mix="short", arrival="poisson", rate=rate, seed=21)
        for name, cfg in [("continuous_b32", dict(policy="continuous", max_batch=32)),
                          ("static_b32_w0", dict(policy="static", max_batch=32, window_ms=0)),
                          ("static_b32_w25", dict(policy="static", max_batch=32, window_ms=25)),
                          ("static_b32_w100", dict(policy="static", max_batch=32, window_ms=100)),
                          ("static_b32_w400", dict(policy="static", max_batch=32, window_ms=400))]:
            out = out_dir / "open" / f"{name}_r{rate:g}.json"
            if out.exists():
                continue
            print(f"[batching-open] {name} rate={rate}", flush=True)
            spec = WorkerSpec(port=port, **cfg)
            with Cluster([spec], router_policy=None) as cl:
                _run(spec.url, tr, out, {"exp": "batching_open", "config": name, "rate": rate, **cfg}, "open",
                     worker_api=True, stats_urls=[spec.url + "/stats"])


# --------------------------------------------------------------------------- #
# E4 - scaling with replicated workers
# --------------------------------------------------------------------------- #
def exp_scaling() -> None:
    out_dir = RES / "scaling"
    base = trace("balanced_closed_256", n=256, mix="balanced", arrival="closed", seed=11)
    levels = [4, 16, 32, 64, 128]
    for k in [1, 2, 3]:
        todo = [c for c in levels if not (out_dir / f"replicas{k}_c{c}.json").exists()]
        if not todo:
            continue
        specs = [WorkerSpec(port=8301 + i, policy="continuous", max_batch=32) for i in range(k)]
        print(f"[scaling] replicas={k}", flush=True)
        with Cluster(specs, router_policy="least_outstanding", router_port=8300) as cl:
            for c in todo:
                n = min(max(3 * c, 48), 256)
                _run(cl.router_url, base[:n], out_dir / f"replicas{k}_c{c}.json",
                     {"exp": "scaling", "replicas": k, "concurrency": c}, "closed", c,
                     stats_urls=[u + "/stats" for u in cl.worker_urls])


# --------------------------------------------------------------------------- #
# E5 - routing policies
# --------------------------------------------------------------------------- #
CLUSTERS = {
    # three workers that differ in precision (speed) and memory headroom
    "hetero": [
        dict(precision="fp16", max_batch=32, kv_budget_tokens=65536, name="A-fp16"),
        dict(precision="int8", max_batch=32, kv_budget_tokens=65536, name="B-int8"),
        dict(precision="fp16", max_batch=8, kv_budget_tokens=8192, name="C-fp16-small"),
    ],
    "homo": [
        dict(precision="fp16", max_batch=32, kv_budget_tokens=65536, name="A-fp16"),
        dict(precision="fp16", max_batch=32, kv_budget_tokens=65536, name="B-fp16"),
        dict(precision="fp16", max_batch=32, kv_budget_tokens=65536, name="C-fp16"),
    ],
}
ROUTING_WORKLOADS = {
    "poisson": dict(n=160, mix="balanced", arrival="poisson", rate=1.5, seed=31),
    "bursty": dict(n=160, mix="balanced", arrival="bursty", rate=1.5, burst=6.0, seed=32),
    "heavy_tail": dict(n=160, mix="heavy_tail", arrival="poisson", rate=1.5, seed=33),
}
POLICY_LIST = ["random", "round_robin", "least_outstanding", "gpu_aware"]


def exp_routing(clusters=("hetero", "homo"), workloads=tuple(ROUTING_WORKLOADS), reps: int = 1) -> None:
    out_dir = RES / "routing"
    for cname in clusters:
        specs = [WorkerSpec(port=8401 + i, **w) for i, w in enumerate(CLUSTERS[cname])]
        todo = [(w, p, r) for w in workloads for p in POLICY_LIST for r in range(reps)
                if not (out_dir / f"{cname}_{w}_{p}_rep{r}.json").exists()]
        if not todo:
            continue
        print(f"[routing] cluster={cname}", flush=True)
        cl = Cluster(specs, router_policy=None, router_port=8400)
        with cl:
            for w, p, r in todo:
                tr = trace(f"routing_{w}", **ROUTING_WORKLOADS[w])
                cl.start_router(policy=p)
                _run(cl.router_url, tr, out_dir / f"{cname}_{w}_{p}_rep{r}.json",
                     {"exp": "routing", "cluster": cname, "workload": w, "policy": p, "rep": r,
                      "workers": CLUSTERS[cname]}, "open", stats_urls=[u + "/stats" for u in cl.worker_urls])
                cl.stop_router()
                _drain(cl.worker_urls)


def _drain(urls, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        busy = 0
        for u in urls:
            try:
                s = httpx.get(u + "/stats", timeout=2).json()
                busy += s["active"] + s["queue_depth"]
            except Exception:
                pass
        if busy == 0:
            return
        time.sleep(1)


# --------------------------------------------------------------------------- #
# E6 - fault tolerance: kill one worker mid-run
# --------------------------------------------------------------------------- #
def exp_fault() -> None:
    out = RES / "fault" / "kill_worker.json"
    if out.exists():
        return
    specs = [WorkerSpec(port=8501 + i, max_batch=32, name=f"W{i}") for i in range(3)]
    tr = trace("fault_poisson", n=150, mix="short", arrival="poisson", rate=3.0, seed=41)
    kill_at = 20.0
    with Cluster(specs, router_policy="gpu_aware", router_port=8500) as cl:

        async def go():
            async def killer():
                await asyncio.sleep(kill_at)
                cl.kill_worker(specs[2].url)
                print(f"  killed {specs[2].url} at t={kill_at}s", flush=True)

            k = asyncio.create_task(killer())
            res = await run_trace(cl.router_url, tr, mode="open", stats_urls=[u + "/stats" for u in cl.worker_urls])
            await k
            return res

        results, samples, makespan = asyncio.run(go())
        router_stats = httpx.get(cl.router_url + "/stats").json()
    s = summarize(results, makespan, samples)
    s["router_retries"] = router_stats["retries"]
    save_results(out, results, samples, s, {"exp": "fault", "kill_at": kill_at, "killed": specs[2].url})
    print(f"  fault: ok={s['n_ok']}/{s['n']} retries={s['router_retries']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["batching", "scaling", "routing", "fault", "all"])
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    if a.exp in ("batching", "all"):
        exp_batching()
    if a.exp in ("scaling", "all"):
        exp_scaling()
    if a.exp in ("routing", "all"):
        exp_routing(reps=a.reps)
    if a.exp in ("fault", "all"):
        exp_fault()


if __name__ == "__main__":
    main()
