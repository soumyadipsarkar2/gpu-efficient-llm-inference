"""Router / API gateway in front of a pool of model workers.

* Exposes ``POST /v1/completions`` (OpenAI-style fields: ``prompt``,
  ``max_tokens``, ``temperature``, ``stream``) and proxies it to one worker.
* Polls every worker's ``/stats`` in the background (default every 100 ms).
  A worker that fails ``fail_threshold`` consecutive polls, or a request, is
  marked unhealthy and skipped until it answers again.
* Keeps router-local in-flight counters that are updated synchronously, so
  routing never acts on stale queue lengths.
* Retries a request on another worker if the chosen one fails before sending
  its first byte.

Run:  python -m llmserve.router --workers http://127.0.0.1:8001,http://127.0.0.1:8002 --policy gpu_aware
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .policies import ReqInfo, WorkerView, make_policy


@dataclass
class WorkerState:
    url: str
    view: WorkerView
    fails: int = 0
    last_poll: float = 0.0
    dispatched_since_poll: int = 0  # KV tokens reserved by requests sent after the last poll
    stats: dict = field(default_factory=dict)
    routed: int = 0


class Router:
    def __init__(self, urls: list[str], policy: str, tokenizer: str | None, poll_interval: float = 0.1,
                 fail_threshold: int = 2, seed: int = 0):
        self.workers = [WorkerState(u.rstrip("/"), WorkerView(name=u)) for u in urls]
        self.policy = make_policy(policy, seed=seed)
        self.poll_interval = poll_interval
        self.fail_threshold = fail_threshold
        self.tok = None
        if tokenizer:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.client: httpx.AsyncClient | None = None
        self.decisions: list[dict] = []
        self.n_retries = 0

    # --------------------------------------------------------------- polling
    async def poll_forever(self) -> None:
        while True:
            await asyncio.gather(*(self.poll(w) for w in self.workers))
            await asyncio.sleep(self.poll_interval)

    async def poll(self, w: WorkerState) -> None:
        try:
            r = await self.client.get(f"{w.url}/stats", timeout=1.0)
            r.raise_for_status()
            s = r.json()
        except Exception:
            w.fails += 1
            if w.fails >= self.fail_threshold:
                w.view.healthy = False
            return
        w.fails = 0
        w.view.healthy = True
        w.stats = s
        w.last_poll = time.perf_counter()
        w.dispatched_since_poll = 0
        v = w.view
        v.name = s.get("name", w.url)
        v.max_batch = s.get("max_batch", v.max_batch)
        v.kv_budget_tokens = s.get("kv_budget_tokens", v.kv_budget_tokens)
        v.kv_free_tokens = s.get("kv_free_tokens", v.kv_free_tokens) - s.get("queued_tokens", 0)
        v.step_a = s.get("step_time_a")
        v.step_c = s.get("step_time_c")
        v.prefill_s_per_token = s.get("prefill_s_per_token")
        v.latency_per_token_ewma = s.get("latency_per_token_ewma")

    # --------------------------------------------------------------- routing
    def count_prompt_tokens(self, body: dict) -> int:
        if "prompt_ids" in body:
            return len(body["prompt_ids"])
        if self.tok is None:
            text = body.get("prompt") or json.dumps(body.get("messages", ""))
            return max(1, len(text) // 4)
        if "messages" in body:
            return len(self.tok.apply_chat_template(body["messages"], add_generation_prompt=True))
        if body.get("chat", True):
            return len(self.tok.apply_chat_template([{"role": "user", "content": body.get("prompt", "")}],
                                                    add_generation_prompt=True))
        return len(self.tok.encode(body.get("prompt", "")))

    def pick(self, info: ReqInfo, exclude: set[int]) -> int:
        views = []
        for i, w in enumerate(self.workers):
            v = w.view
            v.healthy = v.healthy and i not in exclude
            views.append(v)
        try:
            return self.policy.choose(info, views)
        finally:
            for i in exclude:
                self.workers[i].view.healthy = self.workers[i].fails < self.fail_threshold

    def _acquire(self, i: int, info: ReqInfo) -> None:
        w = self.workers[i]
        w.view.inflight += 1
        w.view.inflight_tokens += info.max_tokens
        w.view.kv_free_tokens -= info.reserve
        w.dispatched_since_poll += info.reserve
        w.routed += 1

    def _release(self, i: int, info: ReqInfo) -> None:
        w = self.workers[i]
        w.view.inflight -= 1
        w.view.inflight_tokens -= info.max_tokens

    # ------------------------------------------------------------------ app
    def app(self) -> FastAPI:
        app = FastAPI(title="llmserve router")

        @app.on_event("startup")
        async def _start():
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=2.0),
                                            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512))
            await asyncio.gather(*(self.poll(w) for w in self.workers))
            app.state.poller = asyncio.create_task(self.poll_forever())

        @app.on_event("shutdown")
        async def _stop():
            app.state.poller.cancel()
            await self.client.aclose()

        @app.get("/health")
        async def health():
            n = sum(w.view.healthy for w in self.workers)
            return JSONResponse({"healthy_workers": n, "total": len(self.workers)}, status_code=200 if n else 503)

        @app.get("/stats")
        async def stats():
            return {
                "policy": self.policy.name,
                "retries": self.n_retries,
                "workers": [
                    {"url": w.url, "healthy": w.view.healthy, "inflight": w.view.inflight, "routed": w.routed,
                     "stats": w.stats}
                    for w in self.workers
                ],
            }

        @app.post("/admin/reset")
        async def reset():
            self.decisions.clear()
            for w in self.workers:
                w.routed = 0
            return {"ok": True}

        @app.post("/v1/completions")
        async def completions(request: Request):
            body = await request.json()
            body.setdefault("max_tokens", 128)
            info = ReqInfo(self.count_prompt_tokens(body), int(body["max_tokens"]))
            stream = bool(body.get("stream", False))
            tried: set[int] = set()
            last_err = None
            for _attempt in range(len(self.workers)):
                try:
                    i = self.pick(info, tried)
                except RuntimeError:
                    break
                tried.add(i)
                w = self.workers[i]
                self._acquire(i, info)
                t_route = time.perf_counter()
                try:
                    if stream:
                        req = self.client.build_request("POST", f"{w.url}/generate", json=body)
                        resp = await self.client.send(req, stream=True)
                        if resp.status_code != 200:
                            err = await resp.aread()
                            await resp.aclose()
                            self._release(i, info)
                            if resp.status_code in (503,):
                                last_err = err
                                continue
                            raise HTTPException(resp.status_code, err.decode(errors="ignore"))

                        async def relay(resp=resp, i=i):
                            try:
                                async for chunk in resp.aiter_raw():
                                    yield chunk
                            finally:
                                await resp.aclose()
                                self._release(i, info)

                        return StreamingResponse(relay(), media_type="text/event-stream",
                                                 headers={"x-worker": w.view.name})
                    r = await self.client.post(f"{w.url}/generate", json=body)
                    self._release(i, info)
                    if r.status_code == 503:
                        last_err = r.text
                        continue
                    if r.status_code != 200:
                        raise HTTPException(r.status_code, r.text)
                    out = r.json()
                    out["router_s"] = time.perf_counter() - t_route
                    return out
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                    self._release(i, info)
                    w.fails = self.fail_threshold
                    w.view.healthy = False
                    self.n_retries += 1
                    last_err = str(e)
                    continue
            raise HTTPException(503, f"no worker could serve the request: {last_err}")

        return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="llmserve router")
    ap.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    ap.add_argument("--policy", default="gpu_aware",
                    choices=["random", "round_robin", "least_outstanding", "gpu_aware"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--poll-interval", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    r = Router(a.workers.split(","), a.policy, a.tokenizer, a.poll_interval, seed=a.seed)
    print(json.dumps({"event": "router_ready", "port": a.port, "policy": a.policy}), flush=True)
    uvicorn.run(r.app(), host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
