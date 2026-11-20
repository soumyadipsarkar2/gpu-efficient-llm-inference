"""Request-routing policies.

Policies are pure functions of a list of :class:`WorkerView` snapshots and the
incoming request, so the same code drives both the live HTTP router
(:mod:`llmserve.router`) and the discrete-event simulator (:mod:`llmserve.sim`).

A WorkerView mixes two kinds of signal:

* **router-local** counters (``inflight``, ``inflight_tokens``), updated the
  instant the router dispatches or completes a request, so always fresh;
* **polled** worker telemetry (KV-cache headroom, batch limit, fitted step-time
  model, prefill speed, recent latency), refreshed every ``poll_interval`` and
  therefore up to that much stale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ReqInfo:
    prompt_tokens: int
    max_tokens: int

    @property
    def reserve(self) -> int:
        return self.prompt_tokens + self.max_tokens


@dataclass
class WorkerView:
    name: str
    healthy: bool = True
    inflight: int = 0
    inflight_tokens: int = 0  # sum of max_tokens of in-flight requests
    max_batch: int = 32
    kv_budget_tokens: int = 65536
    kv_free_tokens: int = 65536  # polled headroom minus reservations dispatched since the poll
    step_a: float | None = None  # decode step time model t(b) = a + c*b  (seconds)
    step_c: float | None = None
    prefill_s_per_token: float | None = None
    latency_per_token_ewma: float | None = None
    mem_free_bytes: int | None = None
    extra: dict = field(default_factory=dict)


class Policy:
    name = "base"

    def choose(self, req: ReqInfo, workers: list[WorkerView]) -> int:
        raise NotImplementedError


def _healthy(workers: list[WorkerView]) -> list[int]:
    idx = [i for i, w in enumerate(workers) if w.healthy]
    if not idx:
        raise RuntimeError("no healthy workers")
    return idx


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, req, workers):
        return self.rng.choice(_healthy(workers))


class RoundRobinPolicy(Policy):
    name = "round_robin"

    def __init__(self):
        self.i = -1

    def choose(self, req, workers):
        idx = _healthy(workers)
        n = len(workers)
        for _ in range(n):
            self.i = (self.i + 1) % n
            if self.i in idx:
                return self.i
        return idx[0]


class LeastOutstandingPolicy(Policy):
    """Join-the-shortest-queue on router-local in-flight counts."""

    name = "least_outstanding"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, req, workers):
        idx = _healthy(workers)
        best = min(workers[i].inflight for i in idx)
        return self.rng.choice([i for i in idx if workers[i].inflight == best])


class GpuAwarePolicy(Policy):
    """Pick the worker with the lowest *estimated completion time*.

    For worker *i* with ``n`` requests in flight, batch limit ``B``, fitted decode
    step time ``t_i(b) = a_i + c_i*b`` and prefill cost ``p_i`` per token::

        batch      = min(n + 1, B)
        queue_wait = max(0, n + 1 - B) / B * avg_len_i * t_i(B)     # slots free up as requests finish
        mem_wait   = avg_len_i * t_i(B) / B   if KV headroom < request reservation
        est_i      = queue_wait + mem_wait + p_i * prompt + max_tokens * t_i(batch)

    Workers that can never fit the request (budget too small) are excluded. The
    step-time model captures *model configuration* (precision, model size,
    hardware), ``n`` captures queue depth and active batch size, KV headroom
    captures free GPU memory, and the fitted model is re-estimated from recent
    step latencies, so all signals listed in the proposal enter the score.
    """

    name = "gpu_aware"

    def __init__(self, default_step: float = 0.03, default_prefill: float = 2e-4, seed: int = 0):
        self.default_step = default_step
        self.default_prefill = default_prefill
        self.rng = random.Random(seed)

    def estimate(self, req: ReqInfo, w: WorkerView) -> float:
        if req.reserve > w.kv_budget_tokens:
            return float("inf")
        a = w.step_a if w.step_a is not None else self.default_step
        c = w.step_c if w.step_c is not None else 0.0
        p = w.prefill_s_per_token if w.prefill_s_per_token is not None else self.default_prefill
        B = max(w.max_batch, 1)
        n = w.inflight
        avg_len = (w.inflight_tokens / n) if n else req.max_tokens

        def t(b: int) -> float:
            return a + c * b

        batch = min(n + 1, B)
        queue_wait = max(0, n + 1 - B) / B * avg_len * t(B)
        mem_wait = 0.0
        if w.kv_free_tokens < req.reserve:
            mem_wait = avg_len * t(B) / B * max(1.0, (req.reserve - w.kv_free_tokens) / max(req.reserve, 1))
        return queue_wait + mem_wait + p * req.prompt_tokens + req.max_tokens * t(batch)

    def choose(self, req, workers):
        idx = _healthy(workers)
        scores = [(self.estimate(req, workers[i]), i) for i in idx]
        best = min(s for s, _ in scores)
        if best == float("inf"):
            return LeastOutstandingPolicy().choose(req, workers)
        ties = [i for s, i in scores if s <= best * 1.0001]
        return self.rng.choice(ties)


POLICIES = {
    "random": RandomPolicy,
    "round_robin": RoundRobinPolicy,
    "least_outstanding": LeastOutstandingPolicy,
    "gpu_aware": GpuAwarePolicy,
}


def make_policy(name: str, **kw) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name}; choose from {list(POLICIES)}")
    cls = POLICIES[name]
    try:
        return cls(**kw)
    except TypeError:
        return cls()
