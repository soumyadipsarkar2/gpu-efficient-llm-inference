"""Asynchronous load generator.

Replays a trace against a router (``/v1/completions``) or a single worker
(``/generate``) using streaming, and records per-request client-side timings:

    ttft  - send -> first streamed token
    e2e   - send -> final event
    tpot  - (e2e - ttft) / (tokens - 1), time per output token after the first

While the run is in progress it samples every worker's ``/stats`` endpoint to
record device memory, KV usage and engine busy fraction.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx
import numpy as np

from .workload import TraceRequest


@dataclass
class Result:
    idx: int
    cls: str
    prompt_tokens: int
    max_tokens: int
    t_arrival: float  # scheduled offset
    t_send: float  # actual offset
    ttft: float | None
    e2e: float | None
    tokens: int
    worker: str | None
    ok: bool
    error: str | None = None


async def _one(client: httpx.AsyncClient, url: str, r: TraceRequest, t0: float, ignore_eos: bool,
               worker_api: bool) -> Result:
    body = {"prompt": r.prompt, "max_tokens": r.max_tokens, "stream": True, "ignore_eos": ignore_eos,
            "temperature": 0.0}
    endpoint = f"{url}/generate" if worker_api else f"{url}/v1/completions"
    ts = time.perf_counter()
    ttft = None
    n = 0
    worker = None
    try:
        async with client.stream("POST", endpoint, json=body) as resp:
            if resp.status_code != 200:
                txt = (await resp.aread()).decode(errors="ignore")[:200]
                return Result(r.idx, r.cls, r.prompt_tokens, r.max_tokens, r.t, ts - t0, None, None, 0, None,
                              False, f"HTTP {resp.status_code}: {txt}")
            final = {}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("done"):
                    final = ev
                    break
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - ts
            e2e = time.perf_counter() - ts
            worker = final.get("worker")
            n = final.get("completion_tokens", n)
            return Result(r.idx, r.cls, r.prompt_tokens, r.max_tokens, r.t, ts - t0, ttft, e2e, n, worker,
                          final.get("finish_reason") in ("stop", "length"))
    except Exception as e:  # network error, worker crash, ...
        return Result(r.idx, r.cls, r.prompt_tokens, r.max_tokens, r.t, ts - t0, None, None, 0, worker, False,
                      f"{type(e).__name__}: {e}")


async def _monitor(client: httpx.AsyncClient, stats_urls: list[str], stop: asyncio.Event, t0: float,
                   samples: list, interval: float = 0.5) -> None:
    while not stop.is_set():
        for u in stats_urls:
            try:
                s = (await client.get(u, timeout=1.0)).json()
                samples.append({"t": time.perf_counter() - t0, "url": u, "active": s.get("active"),
                                "queue": s.get("queue_depth"), "mem": s.get("mem_allocated_bytes"),
                                "mem_reserved": s.get("mem_reserved_bytes"), "busy": s.get("busy_fraction"),
                                "kv_physical": s.get("kv_physical_tokens"), "kv_useful": s.get("kv_useful_tokens"),
                                "kv_reserved": s.get("kv_reserved_tokens"), "tokens_out": s.get("tokens_out")})
            except Exception:
                pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_trace(url: str, trace: list[TraceRequest], mode: str = "open", concurrency: int = 8,
                    ignore_eos: bool = True, worker_api: bool = False, stats_urls: list[str] | None = None,
                    time_scale: float = 1.0) -> tuple[list[Result], list[dict], float]:
    """Replay ``trace``. ``mode='open'`` honours arrival offsets (scaled by
    ``time_scale``); ``mode='closed'`` keeps ``concurrency`` requests in flight."""
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=1024)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=5.0), limits=limits) as client:
        samples: list[dict] = []
        stop = asyncio.Event()
        t0 = time.perf_counter()
        mon = asyncio.create_task(_monitor(client, stats_urls or [], stop, t0, samples))
        results: list[Result] = []
        if mode == "open":

            async def delayed(r: TraceRequest):
                delay = r.t * time_scale - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                return await _one(client, url, r, t0, ignore_eos, worker_api)

            results = list(await asyncio.gather(*(delayed(r) for r in trace)))
        else:
            it = iter(trace)

            async def user():
                for r in it:
                    results.append(await _one(client, url, r, t0, ignore_eos, worker_api))

            await asyncio.gather(*(user() for _ in range(concurrency)))
        makespan = time.perf_counter() - t0
        stop.set()
        await mon
    return results, samples, makespan


def summarize(results: list[Result], makespan: float, samples: list[dict] | None = None) -> dict:
    ok = [r for r in results if r.ok]
    out = {"n": len(results), "n_ok": len(ok), "n_err": len(results) - len(ok), "makespan_s": makespan}
    if not ok:
        return out
    e2e = np.array([r.e2e for r in ok])
    ttft = np.array([r.ttft for r in ok if r.ttft is not None])
    tpot = np.array([(r.e2e - r.ttft) / max(r.tokens - 1, 1) for r in ok if r.ttft is not None and r.tokens > 1])
    toks = sum(r.tokens for r in ok)
    out.update({
        "throughput_tok_s": toks / makespan,
        "throughput_req_s": len(ok) / makespan,
        "e2e_p50": float(np.percentile(e2e, 50)), "e2e_p95": float(np.percentile(e2e, 95)),
        "e2e_p99": float(np.percentile(e2e, 99)), "e2e_mean": float(e2e.mean()),
        "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p95": float(np.percentile(ttft, 95)),
        "ttft_p99": float(np.percentile(ttft, 99)), "ttft_mean": float(ttft.mean()),
        "tpot_p50": float(np.percentile(tpot, 50)) if len(tpot) else None,
        "tpot_p95": float(np.percentile(tpot, 95)) if len(tpot) else None,
        "tokens": toks,
    })
    if samples:
        mem = [s["mem"] for s in samples if s.get("mem")]
        busy = [s["busy"] for s in samples if s.get("busy") is not None]
        act = [s["active"] for s in samples if s.get("active") is not None]
        kvp = [s["kv_physical"] for s in samples if s.get("kv_physical")]
        kvu = [s["kv_useful"] for s in samples if s.get("kv_useful")]
        out.update({
            "mem_peak_bytes": max(mem) if mem else None,
            "busy_last": busy[-1] if busy else None,
            "active_mean": float(np.mean(act)) if act else None,
            "kv_padding_overhead": (sum(kvp) / sum(kvu) - 1) if kvu and sum(kvu) else None,
        })
    workers = {}
    for r in ok:
        workers[r.worker] = workers.get(r.worker, 0) + 1
    out["per_worker"] = workers
    return out


def save_results(path, results: list[Result], samples: list[dict], summary: dict, meta: dict) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": meta, "summary": summary, "results": [asdict(r) for r in results],
                             "samples": samples}, indent=None))
