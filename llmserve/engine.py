"""Inference engine with three batching policies.

The engine owns one model replica and runs a single scheduling loop in a
background thread (the "GPU loop"). Requests are submitted from any thread and
report tokens back through callbacks.

Batching policies
-----------------
``serial``      one request at a time (batch size 1). Baseline.
``static``      request-level dynamic batching: wait until ``max_batch`` requests
                are queued or the oldest has waited ``window_ms``, then run that
                batch until its *longest* member finishes. New arrivals wait for
                the whole batch to drain (this is what ``model.generate`` on a
                padded batch does).
``continuous``  iteration-level batching (Orca / vLLM style): before every decode
                step, queued requests are prefilled and merged into the running
                batch if there is a free batch slot and KV-cache budget; finished
                requests leave immediately.

KV-cache layout
---------------
All running sequences share one Hugging Face ``DynamicCache`` of shape
``[B, kv_heads, T, head_dim]`` per layer plus a 2-D attention mask ``[B, T]``.
New sequences are prefilled as a right-padded group (so no query row is fully
masked, which would produce NaNs in SDPA), then left-padded with zero KV to the
running batch's length and concatenated along the batch axis. Positions are
passed explicitly so RoPE sees each sequence's true positions regardless of
padding. When sequences leave, rows are dropped and leading all-padding columns
are trimmed.
"""

from __future__ import annotations

import itertools
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import DynamicCache

from . import device as dev
from .model_loader import LoadedModel, eos_ids

_rid = itertools.count()


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_new_tokens: int = 128
    temperature: float = 0.0
    ignore_eos: bool = False
    seed: int | None = None
    rid: str = field(default_factory=lambda: f"r{next(_rid)}")
    on_token: Callable[[int, float], None] | None = None
    on_finish: Callable[[dict], None] | None = None
    # filled in by the engine
    t_submit: float = 0.0
    t_admit: float | None = None
    t_first: float | None = None
    t_done: float | None = None
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    cancelled: bool = False

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def reserve_tokens(self) -> int:
        return self.prompt_len + self.max_new_tokens


@dataclass
class EngineConfig:
    policy: str = "continuous"  # serial | static | continuous
    max_batch: int = 32
    window_ms: float = 20.0  # static policy only
    kv_budget_tokens: int = 65536  # admission limit on reserved KV tokens
    max_prefill_tokens: int = 4096  # cap on padded tokens prefilled per iteration
    max_queue: int = 4096


class _Running:
    """Book-keeping for the running batch."""

    def __init__(self) -> None:
        self.reqs: list[GenRequest] = []
        self.cache: DynamicCache | None = None
        self.mask: torch.Tensor | None = None  # [B, T] long
        self.next_pos: list[int] = []
        self.last_tok: list[int] = []

    def __len__(self) -> int:
        return len(self.reqs)

    @property
    def T(self) -> int:
        return 0 if self.mask is None else self.mask.shape[1]


class Engine:
    def __init__(self, lm: LoadedModel, cfg: EngineConfig):
        if cfg.policy not in ("serial", "static", "continuous"):
            raise ValueError(cfg.policy)
        if cfg.policy == "serial":
            cfg.max_batch = 1
            cfg.window_ms = 0.0
        self.lm = lm
        self.cfg = cfg
        self.model = lm.model
        self.base = lm.model.get_decoder() if hasattr(lm.model, "get_decoder") else lm.model.model
        self.lm_head = lm.model.lm_head
        self.device = lm.device
        self.eos = eos_ids(lm)
        self.pad_id = lm.tokenizer.pad_token_id or 0
        self.inbox: "queue.Queue[GenRequest]" = queue.Queue()
        self.waiting: deque[GenRequest] = deque()
        self.run = _Running()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # metrics
        self.started_at = time.perf_counter()
        self.busy_s = 0.0
        self.n_submitted = 0
        self.n_finished = 0
        self.n_rejected = 0
        self.tokens_out = 0
        self.tokens_prefilled = 0
        self.decode_steps = 0
        self.prefill_calls = 0
        self.step_samples: deque[tuple[int, int, float]] = deque(maxlen=512)  # (batch, T, seconds)
        self.prefill_samples: deque[tuple[int, float]] = deque(maxlen=256)  # (tokens, seconds)
        self.latency_ewma: float | None = None  # seconds per output token, end-to-end
        self.e2e_ewma: float | None = None
        self.ttft_ewma: float | None = None
        self.batch_hist: dict[int, int] = {}
        self.physical_kv_tokens = 0
        self.useful_kv_tokens = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ API
    def start(self) -> "Engine":
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.inbox.put(None)  # type: ignore[arg-type]
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, req: GenRequest) -> bool:
        if self.queue_depth() >= self.cfg.max_queue:
            self.n_rejected += 1
            return False
        if req.reserve_tokens > self.cfg.kv_budget_tokens:
            self.n_rejected += 1
            req.finish_reason = "too_long"
            if req.on_finish:
                req.on_finish(self._summary(req))
            return False
        req.t_submit = time.perf_counter()
        self.n_submitted += 1
        self.inbox.put(req)
        return True

    def queue_depth(self) -> int:
        return self.inbox.qsize() + len(self.waiting)

    def reserved_kv_tokens(self) -> int:
        return sum(r.reserve_tokens for r in self.run.reqs)

    def stats(self) -> dict:
        now = time.perf_counter()
        wall = max(now - self.started_at, 1e-9)
        a, c = self.step_model()
        p = self.prefill_model()
        waiting = list(self.waiting)
        return {
            "policy": self.cfg.policy,
            "max_batch": self.cfg.max_batch,
            "queue_depth": self.queue_depth(),
            "active": len(self.run),
            "queued_tokens": sum(r.reserve_tokens for r in waiting),
            "kv_budget_tokens": self.cfg.kv_budget_tokens,
            "kv_reserved_tokens": self.reserved_kv_tokens(),
            "kv_free_tokens": self.cfg.kv_budget_tokens - self.reserved_kv_tokens(),
            "kv_physical_tokens": self.physical_kv_tokens,
            "kv_useful_tokens": self.useful_kv_tokens,
            "kv_bytes_per_token": self.lm.kv_bytes_per_token,
            "mem_allocated_bytes": dev.memory_allocated(self.device),
            "mem_reserved_bytes": dev.memory_reserved(self.device),
            "gpu_util_pct": dev.gpu_utilization(self.device),
            "busy_fraction": self.busy_s / wall,
            "submitted": self.n_submitted,
            "finished": self.n_finished,
            "rejected": self.n_rejected,
            "tokens_out": self.tokens_out,
            "tokens_prefilled": self.tokens_prefilled,
            "decode_steps": self.decode_steps,
            "prefill_calls": self.prefill_calls,
            "step_time_a": a,
            "step_time_c": c,
            "prefill_s_per_token": p,
            "latency_per_token_ewma": self.latency_ewma,
            "e2e_ewma": self.e2e_ewma,
            "ttft_ewma": self.ttft_ewma,
            "batch_hist": dict(self.batch_hist),
            "uptime_s": wall,
            "last_error": self.last_error,
        }

    def step_model(self) -> tuple[float | None, float | None]:
        """Least-squares fit of decode step time t(b) = a + c*b over recent steps."""
        s = list(self.step_samples)
        if len(s) < 4:
            return (s[-1][2] if s else None), None
        n = len(s)
        xs = [b for b, _, _ in s]
        ys = [t for _, _, t in s]
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        if vx < 1e-9:
            return my, None
        c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
        c = max(c, 0.0)
        return my - c * mx, c

    def prefill_model(self) -> float | None:
        s = list(self.prefill_samples)
        if not s:
            return None
        return sum(t for _, t in s) / max(sum(n for n, _ in s), 1)

    # ------------------------------------------------------------ main loop
    def _loop(self) -> None:
        torch.set_grad_enabled(False)
        while not self._stop.is_set():
            try:
                self._drain_inbox(block=len(self.run) == 0 and not self.waiting)
                if self._stop.is_set():
                    break
                self._drop_cancelled()
                if self.cfg.policy == "continuous":
                    self._admit_continuous()
                else:
                    self._admit_static()
                if len(self.run):
                    self._decode_step()
            except Exception as e:  # keep the worker alive; fail the running batch
                import traceback

                self.last_error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                for r in self.run.reqs:
                    self._finish(r, "error")
                self.run = _Running()

    def _drain_inbox(self, block: bool, timeout: float | None = 0.05) -> None:
        try:
            if block:
                item = self.inbox.get(timeout=timeout)
                if item is not None:
                    self.waiting.append(item)
            while True:
                item = self.inbox.get_nowait()
                if item is not None:
                    self.waiting.append(item)
        except queue.Empty:
            pass

    def _drop_cancelled(self) -> None:
        if any(r.cancelled for r in self.waiting):
            for r in [r for r in self.waiting if r.cancelled]:
                self.waiting.remove(r)
                self._finish(r, "cancelled")
        idx = [i for i, r in enumerate(self.run.reqs) if r.cancelled]
        if idx:
            for i in idx:
                self._finish(self.run.reqs[i], "cancelled")
            self._remove_rows(idx)

    # ------------------------------------------------------------ admission
    def _take_admissible(self, slots: int) -> list[GenRequest]:
        free_kv = self.cfg.kv_budget_tokens - self.reserved_kv_tokens()
        out: list[GenRequest] = []
        padded_max = 0
        while self.waiting and len(out) < slots:
            r = self.waiting[0]
            if r.reserve_tokens > free_kv:
                break  # FIFO: do not let short requests overtake (no starvation)
            new_max = max(padded_max, r.prompt_len)
            if out and new_max * (len(out) + 1) > self.cfg.max_prefill_tokens:
                break
            self.waiting.popleft()
            free_kv -= r.reserve_tokens
            padded_max = new_max
            out.append(r)
        return out

    def _admit_continuous(self) -> None:
        slots = self.cfg.max_batch - len(self.run)
        while slots > 0 and self.waiting:
            group = self._take_admissible(slots)
            if not group:
                break
            self._prefill_and_merge(group)
            slots = self.cfg.max_batch - len(self.run)

    def _admit_static(self) -> None:
        if len(self.run) or not self.waiting:
            return
        oldest_wait = (time.perf_counter() - self.waiting[0].t_submit) * 1000.0
        if len(self.waiting) < self.cfg.max_batch and oldest_wait < self.cfg.window_ms:
            # wait for more arrivals, but no longer than the window
            remaining = (self.cfg.window_ms - oldest_wait) / 1000.0
            self._drain_inbox(block=True, timeout=max(remaining, 0.0005))
            oldest_wait = (time.perf_counter() - self.waiting[0].t_submit) * 1000.0
            if len(self.waiting) < self.cfg.max_batch and oldest_wait < self.cfg.window_ms:
                return
        slots = self.cfg.max_batch
        while slots > 0 and self.waiting:
            group = self._take_admissible(slots)
            if not group:
                break
            self._prefill_and_merge(group)
            slots = self.cfg.max_batch - len(self.run)

    # ------------------------------------------------------------- compute
    def _timed(self, fn):
        t0 = time.perf_counter()
        out = fn()
        dev.synchronize(self.device)
        dt = time.perf_counter() - t0
        self.busy_s += dt
        return out, dt

    def _prefill_and_merge(self, group: list[GenRequest]) -> None:
        now = time.perf_counter()
        for r in group:
            r.t_admit = now
        n = len(group)
        L = max(r.prompt_len for r in group)
        ids = torch.full((n, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, L), dtype=torch.long)
        for i, r in enumerate(group):
            ids[i, : r.prompt_len] = torch.tensor(r.prompt_ids, dtype=torch.long)
            mask[i, : r.prompt_len] = 1
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(n, L)
        ids, mask, pos = ids.to(self.device), mask.to(self.device), pos.to(self.device)
        last = torch.tensor([r.prompt_len - 1 for r in group], device=self.device)
        cache = DynamicCache()

        def run():
            out = self.base(
                input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True
            )
            h = out.last_hidden_state[torch.arange(n, device=self.device), last]
            return self.lm_head(h).float()

        logits, dt = self._timed(run)
        self.prefill_calls += 1
        self.tokens_prefilled += sum(r.prompt_len for r in group)
        self.prefill_samples.append((n * L, dt))
        toks = self._sample(logits, group)
        self._merge(group, cache, mask, toks)
        t = time.perf_counter()
        for r, tok in zip(group, toks):
            r.t_first = t
            self._emit(r, tok, t)
        self._retire_finished()

    def _merge(self, group, cache: DynamicCache, mask: torch.Tensor, toks: list[int]) -> None:
        run = self.run
        if run.cache is None or len(run) == 0:
            run.cache, run.mask = cache, mask
            run.reqs = list(group)
            run.next_pos = [r.prompt_len for r in group]
            run.last_tok = list(toks)
            self._update_kv_stats()
            return
        T_old, T_new = run.T, mask.shape[1]
        T = max(T_old, T_new)
        old_k, old_v = run.cache.key_cache, run.cache.value_cache
        new_k, new_v = cache.key_cache, cache.value_cache
        for li in range(len(old_k)):
            old_k[li] = torch.cat((_lpad(old_k[li], T), _lpad(new_k[li], T)), dim=0)
            old_v[li] = torch.cat((_lpad(old_v[li], T), _lpad(new_v[li], T)), dim=0)
        run.mask = torch.cat((_lpad_mask(run.mask, T), _lpad_mask(mask, T)), dim=0)
        run.reqs.extend(group)
        run.next_pos.extend(r.prompt_len for r in group)
        run.last_tok.extend(toks)
        self._update_kv_stats()

    def _decode_step(self) -> None:
        run = self.run
        B = len(run)
        ids = torch.tensor(run.last_tok, dtype=torch.long, device=self.device).unsqueeze(1)
        pos = torch.tensor(run.next_pos, dtype=torch.long, device=self.device).unsqueeze(1)
        mask = torch.cat((run.mask, torch.ones((B, 1), dtype=run.mask.dtype, device=self.device)), dim=1)

        def fwd():
            out = self.base(
                input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=run.cache, use_cache=True
            )
            return self.lm_head(out.last_hidden_state[:, -1]).float()

        logits, dt = self._timed(fwd)
        run.mask = mask
        self.decode_steps += 1
        self.step_samples.append((B, run.T, dt))
        self.batch_hist[B] = self.batch_hist.get(B, 0) + 1
        toks = self._sample(logits, run.reqs)
        t = time.perf_counter()
        for i, (r, tok) in enumerate(zip(run.reqs, toks)):
            run.next_pos[i] += 1
            run.last_tok[i] = tok
            self._emit(r, tok, t)
        self._retire_finished()

    def _sample(self, logits: torch.Tensor, reqs: list[GenRequest]) -> list[int]:
        greedy = logits.argmax(dim=-1)
        if all(r.temperature <= 0 for r in reqs):
            return greedy.tolist()
        out = greedy.tolist()
        for i, r in enumerate(reqs):
            if r.temperature > 0:
                p = torch.softmax(logits[i] / r.temperature, dim=-1)
                g = None
                if r.seed is not None:
                    g = torch.Generator(device="cpu").manual_seed(r.seed + len(r.output_ids))
                out[i] = int(torch.multinomial(p.cpu(), 1, generator=g).item())
        return out

    def _emit(self, r: GenRequest, tok: int, t: float) -> None:
        r.output_ids.append(tok)
        self.tokens_out += 1
        if r.on_token:
            r.on_token(tok, t)
        if not r.ignore_eos and tok in self.eos:
            r.finish_reason = "stop"
        elif len(r.output_ids) >= r.max_new_tokens:
            r.finish_reason = "length"

    def _retire_finished(self) -> None:
        idx = [i for i, r in enumerate(self.run.reqs) if r.finish_reason is not None]
        if not idx:
            return
        for i in idx:
            self._finish(self.run.reqs[i], self.run.reqs[i].finish_reason)
        self._remove_rows(idx)

    def _remove_rows(self, idx: list[int]) -> None:
        run = self.run
        keep = [i for i in range(len(run)) if i not in set(idx)]
        if not keep:
            self.run = _Running()
            self._update_kv_stats()
            return
        kt = torch.tensor(keep, device=self.device)
        mask = run.mask.index_select(0, kt)
        # trim leading columns that are padding for every remaining row
        nz = mask.any(dim=0).nonzero()
        start = int(nz[0].item()) if nz.numel() else 0
        mask = mask[:, start:]
        for li in range(len(run.cache.key_cache)):
            run.cache.key_cache[li] = run.cache.key_cache[li].index_select(0, kt)[:, :, start:].contiguous()
            run.cache.value_cache[li] = run.cache.value_cache[li].index_select(0, kt)[:, :, start:].contiguous()
        run.mask = mask
        run.reqs = [run.reqs[i] for i in keep]
        run.next_pos = [run.next_pos[i] for i in keep]
        run.last_tok = [run.last_tok[i] for i in keep]
        self._update_kv_stats()

    def _update_kv_stats(self) -> None:
        run = self.run
        if run.mask is None or len(run) == 0:
            self.physical_kv_tokens = 0
            self.useful_kv_tokens = 0
            return
        self.physical_kv_tokens = run.mask.shape[0] * run.mask.shape[1]
        self.useful_kv_tokens = sum(run.next_pos)

    # ----------------------------------------------------------- finishing
    def _summary(self, r: GenRequest) -> dict:
        def d(a, b):
            return None if a is None or b is None else b - a

        return {
            "rid": r.rid,
            "prompt_tokens": r.prompt_len,
            "completion_tokens": len(r.output_ids),
            "finish_reason": r.finish_reason,
            "queue_s": d(r.t_submit, r.t_admit),
            "ttft_s": d(r.t_submit, r.t_first),
            "e2e_s": d(r.t_submit, r.t_done),
        }

    def _finish(self, r: GenRequest, reason: str) -> None:
        r.finish_reason = reason
        r.t_done = time.perf_counter()
        self.n_finished += 1
        s = self._summary(r)
        if reason in ("stop", "length") and s["e2e_s"] is not None:
            per_tok = s["e2e_s"] / max(len(r.output_ids), 1)
            self.latency_ewma = _ewma(self.latency_ewma, per_tok)
            self.e2e_ewma = _ewma(self.e2e_ewma, s["e2e_s"])
            if s["ttft_s"] is not None:
                self.ttft_ewma = _ewma(self.ttft_ewma, s["ttft_s"])
        if r.on_finish:
            r.on_finish(s)


def _ewma(old: float | None, x: float, alpha: float = 0.2) -> float:
    return x if old is None or math.isnan(old) else (1 - alpha) * old + alpha * x


def _lpad(x: torch.Tensor, T: int) -> torch.Tensor:
    """Left-pad a KV tensor [B, H, t, D] with zeros along the sequence axis."""
    t = x.shape[2]
    if t == T:
        return x
    pad = torch.zeros((x.shape[0], x.shape[1], T - t, x.shape[3]), dtype=x.dtype, device=x.device)
    return torch.cat((pad, x), dim=2)


def _lpad_mask(m: torch.Tensor, T: int) -> torch.Tensor:
    t = m.shape[1]
    if t == T:
        return m
    pad = torch.zeros((m.shape[0], T - t), dtype=m.dtype, device=m.device)
    return torch.cat((pad, m), dim=1)


def generate_blocking(engine: Engine, prompt_ids: list[int], **kw) -> GenRequest:
    """Convenience helper for tests and offline benchmarks."""
    done = threading.Event()
    req = GenRequest(prompt_ids=prompt_ids, on_finish=lambda s: done.set(), **kw)
    engine.submit(req)
    done.wait()
    return req
