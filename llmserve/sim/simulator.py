"""Discrete-event simulator of a routed, continuously-batched LLM cluster.

Why a simulator: the live experiments run every worker on one physical GPU, so
"more workers" share compute. To study routing on clusters of *independent*
GPUs (and with many random seeds) we replay workloads through a simulator whose
cost model is fitted from real measurements (``bench/profile_engine.py``)::

    decode iteration:  t(B, T_sum) = a + c*B + e*T_sum      (T_sum = cached tokens in batch)
    prefill:           p0 + p1*L per admitted request

Each simulated worker mirrors :class:`llmserve.engine.Engine` with the
``continuous`` policy: FIFO admission bounded by ``max_batch`` and a KV-token
reservation budget, prefill of newly admitted requests at the start of an
iteration (their first token), then one decode step for the whole batch.

The router uses the *same* policy objects as the live router
(:mod:`llmserve.policies`). Router-local in-flight counters are exact; polled
telemetry (KV headroom) is refreshed every ``poll_interval`` seconds.
"""

from __future__ import annotations

import heapq
import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..policies import ReqInfo, WorkerView, make_policy

CLASSES = {
    "short": ((32, 128), (32, 64)),
    "medium": ((256, 512), (64, 128)),
    "long": ((1024, 1536), (128, 256)),
}
MIXES = {
    "short": {"short": 1.0},
    "balanced": {"short": 0.5, "medium": 0.35, "long": 0.15},
    "heavy_tail": {"short": 0.8, "long": 0.2},
}


@dataclass
class Profile:
    """Cost model of one worker.

    If measured grids are present (always the case for :meth:`from_json`) the
    decode step time is interpolated in the measured (batch, mean context) grid
    and prefill time in the measured prompt-length grid, which captures
    non-linear effects such as kernel-selection cliffs and quadratic prefill
    attention. ``a, c, e`` (linear fit) are what the GPU-aware router sees.
    """

    name: str
    a: float
    c: float
    e: float
    p0: float
    p1: float
    max_batch: int = 32
    kv_budget_tokens: int = 65536
    speed: float = 1.0  # multiply all times (emulate slower/faster hardware)
    grid_B: tuple = ()
    grid_T: tuple = ()
    grid_s: tuple = ()  # [len(T)][len(B)]
    pre_L: tuple = ()
    pre_s: tuple = ()

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None, **kw) -> "Profile":
        d = json.loads(Path(path).read_text())
        f = d["fit"]
        Ts = sorted({r["T"] for r in d["decode"]})
        Bs = sorted({r["B"] for r in d["decode"]})
        tab = {(r["B"], r["T"]): r["s"] for r in d["decode"]}
        grid = []
        for T in Ts:
            row = []
            for B in Bs:
                if (B, T) in tab:
                    row.append(tab[(B, T)])
                else:  # grid point skipped for memory: extrapolate from the two largest measured batches
                    have = [(b, tab[(b, T)]) for b in Bs if (b, T) in tab]
                    (b1, s1), (b2, s2) = have[-2], have[-1]
                    row.append(s2 + (s2 - s1) / (b2 - b1) * (B - b2))
            grid.append(tuple(row))
        pre = sorted((r["L"], r["s"]) for r in d["prefill"])
        return cls(name=name or Path(path).stem, a=f["a"], c=f["c"], e=f["e"], p0=max(f["p0"], 0.0), p1=f["p1"],
                   grid_B=tuple(Bs), grid_T=tuple(Ts), grid_s=tuple(grid),
                   pre_L=tuple(L for L, _ in pre), pre_s=tuple(t for _, t in pre), **kw)

    def step(self, B: int, T_sum: int) -> float:
        if self.grid_s:
            T = T_sum / max(B, 1)
            per_T = [_interp(B, self.grid_B, row) for row in self.grid_s]
            return max(_interp(T, self.grid_T, per_T), 1e-4) * self.speed
        return max(self.a + self.c * B + self.e * T_sum, 1e-4) * self.speed

    def prefill(self, L: int) -> float:
        if self.pre_s:
            return max(_interp(L, self.pre_L, self.pre_s), 1e-4) * self.speed
        return max(self.p0 + self.p1 * L, 1e-4) * self.speed


def _interp(x: float, xs, ys) -> float:
    """Piecewise-linear interpolation with linear extrapolation at both ends."""
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class SimReq:
    idx: int
    t_arr: float
    prompt: int
    max_tokens: int
    cls: str = ""
    worker: int = -1
    t_first: float | None = None
    t_done: float | None = None
    generated: int = 0

    @property
    def reserve(self) -> int:
        return self.prompt + self.max_tokens


def synthetic(n: int, mix: str = "balanced", arrival: str = "poisson", rate: float = 2.0, seed: int = 0,
              burst: float = 4.0, mean_on_s: float = 5.0) -> list[SimReq]:
    rng = random.Random(seed)
    classes, weights = zip(*MIXES[mix].items())
    t, out, on = 0.0, [], True
    hi, lo = rate * 2 * burst / (burst + 1), rate * 2 / (burst + 1)
    switch_at = rng.expovariate(1 / mean_on_s)
    while len(out) < n:
        if arrival == "poisson":
            t += rng.expovariate(rate)
        else:  # bursty two-state MMPP with the same mean rate
            dt = rng.expovariate(hi if on else lo)
            if t + dt > switch_at:
                t = switch_at
                on = not on
                switch_at = t + rng.expovariate(1 / mean_on_s)
                continue
            t += dt
        cls = rng.choices(classes, weights)[0]
        (plo, phi), (olo, ohi) = CLASSES[cls]
        out.append(SimReq(len(out), t, rng.randint(plo, phi), rng.randint(olo, ohi), cls))
    return out


class _Worker:
    def __init__(self, i: int, prof: Profile):
        self.i = i
        self.p = prof
        self.queue: deque[SimReq] = deque()
        self.active: list[SimReq] = []
        self.busy = False
        self.reserved = 0
        self.busy_time = 0.0

    def inflight(self) -> int:
        return len(self.queue) + len(self.active)

    def inflight_tokens(self) -> int:
        return sum(r.max_tokens for r in self.queue) + sum(r.max_tokens for r in self.active)


@dataclass
class SimResult:
    reqs: list[SimReq]
    makespan: float
    busy: list[float]
    routed: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        done = [r for r in self.reqs if r.t_done is not None]
        e2e = np.array([r.t_done - r.t_arr for r in done])
        ttft = np.array([r.t_first - r.t_arr for r in done])
        tpot = np.array([(r.t_done - r.t_first) / max(r.generated - 1, 1) for r in done])
        toks = sum(r.generated for r in done)
        return {
            "n": len(self.reqs), "n_done": len(done), "makespan_s": self.makespan,
            "throughput_tok_s": toks / self.makespan, "throughput_req_s": len(done) / self.makespan,
            "e2e_p50": float(np.percentile(e2e, 50)), "e2e_p95": float(np.percentile(e2e, 95)),
            "e2e_p99": float(np.percentile(e2e, 99)), "e2e_mean": float(e2e.mean()),
            "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p95": float(np.percentile(ttft, 95)),
            "ttft_p99": float(np.percentile(ttft, 99)), "tpot_p50": float(np.percentile(tpot, 50)),
            "tpot_p95": float(np.percentile(tpot, 95)),
            "busy": self.busy, "routed": self.routed,
        }


def simulate(profiles: list[Profile], reqs: list[SimReq], policy: str = "gpu_aware", poll_interval: float = 0.1,
             seed: int = 0, typical_ctx: int = 600, stale_inflight: bool = False) -> SimResult:
    """Run one simulation.

    ``stale_inflight=True`` models a router without local counters: queue lengths
    are only learned from worker telemetry at each poll (plus nothing in between),
    which causes "herding" onto whichever worker looked idle at the last poll.
    """
    reqs = [SimReq(r.idx, r.t_arr, r.prompt, r.max_tokens, r.cls) for r in reqs]
    workers = [_Worker(i, p) for i, p in enumerate(profiles)]
    pol = make_policy(policy, seed=seed)
    views = [WorkerView(name=p.name, max_batch=p.max_batch, kv_budget_tokens=p.kv_budget_tokens,
                        kv_free_tokens=p.kv_budget_tokens, step_a=p.a * p.speed,
                        step_c=(p.c + p.e * typical_ctx) * p.speed,
                        prefill_s_per_token=p.prefill(512) / 512)
             for p in profiles]
    routed = [0] * len(workers)
    ev: list = []  # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, payload))
        seq += 1

    for r in reqs:
        push(r.t_arr, "arr", r)
    if poll_interval > 0:
        push(0.0, "poll", None)
    remaining = len(reqs)
    now = 0.0

    def start_iter(w: _Worker, now: float) -> None:
        dt = 0.0
        new = []
        while w.queue and len(w.active) + len(new) < w.p.max_batch:
            r = w.queue[0]
            if w.reserved + r.reserve > w.p.kv_budget_tokens:
                break
            w.queue.popleft()
            w.reserved += r.reserve
            dt += w.p.prefill(r.prompt)
            new.append(r)
        for r in new:
            r.t_first = now + dt
            r.generated = 1
        w.active.extend(new)
        if not w.active:
            w.busy = False
            return
        # requests that finish with their prefill token (max_tokens == 1) skip decode
        B = len(w.active)
        T_sum = sum(r.prompt + r.generated for r in w.active)
        dt += w.p.step(B, T_sum)
        w.busy = True
        w.busy_time += dt
        push(now + dt, "iter", w)

    while ev and remaining > 0:
        now, _, kind, payload = heapq.heappop(ev)
        if kind == "arr":
            r: SimReq = payload
            for i, w in enumerate(workers):
                if not stale_inflight or poll_interval <= 0:
                    views[i].inflight = w.inflight()
                    views[i].inflight_tokens = w.inflight_tokens()
                if poll_interval <= 0:
                    views[i].kv_free_tokens = w.p.kv_budget_tokens - w.reserved - sum(q.reserve for q in w.queue)
            i = pol.choose(ReqInfo(r.prompt, r.max_tokens), views)
            r.worker = i
            routed[i] += 1
            views[i].kv_free_tokens -= r.reserve
            workers[i].queue.append(r)
            if not workers[i].busy:
                start_iter(workers[i], now)
        elif kind == "iter":
            w: _Worker = payload
            still = []
            for r in w.active:
                if r.t_first is not None and r.t_first < now and r.generated >= 1:
                    r.generated += 1
                if r.generated >= r.max_tokens:
                    r.t_done = now
                    w.reserved -= r.reserve
                    remaining -= 1
                else:
                    still.append(r)
            w.active = still
            start_iter(w, now)
        elif kind == "poll":
            for i, w in enumerate(workers):
                views[i].kv_free_tokens = w.p.kv_budget_tokens - w.reserved - sum(q.reserve for q in w.queue)
                if stale_inflight:
                    views[i].inflight = w.inflight()
                    views[i].inflight_tokens = w.inflight_tokens()
            push(now + poll_interval, "poll", None)
    return SimResult(reqs, now, [w.busy_time / max(now, 1e-9) for w in workers], routed)
