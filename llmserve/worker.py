"""Model worker: one model replica + engine behind an HTTP API.

Endpoints
---------
POST /generate   {prompt | prompt_ids | messages, max_tokens, temperature,
                  ignore_eos, stream, chat}
                 stream=true returns Server-Sent Events: one ``{"t": "<text>"}``
                 event per token and a final ``{"done": true, ...}`` event.
GET  /health     liveness + readiness
GET  /stats      JSON runtime state used by the GPU-aware router
GET  /metrics    Prometheus text format

Run:  python -m llmserve.worker --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .engine import Engine, EngineConfig, GenRequest
from .model_loader import load_model


class _Metrics:
    def __init__(self, name: str):
        self.reg = CollectorRegistry()
        lbl = ["worker"]
        self.name = name
        self.requests = Counter("llm_requests_total", "Requests finished", lbl + ["reason"], registry=self.reg)
        self.tokens = Counter("llm_generated_tokens_total", "Tokens generated", lbl, registry=self.reg)
        self.ttft = Histogram(
            "llm_ttft_seconds", "Time to first token", lbl, registry=self.reg,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
        )
        self.e2e = Histogram(
            "llm_e2e_seconds", "End-to-end latency", lbl, registry=self.reg,
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80),
        )
        self.queue = Gauge("llm_queue_depth", "Queued requests", lbl, registry=self.reg)
        self.active = Gauge("llm_active_requests", "Requests in the running batch", lbl, registry=self.reg)
        self.kv_free = Gauge("llm_kv_free_tokens", "Unreserved KV-cache tokens", lbl, registry=self.reg)
        self.mem = Gauge("llm_device_memory_bytes", "Allocated device memory", lbl, registry=self.reg)
        self.busy = Gauge("llm_busy_fraction", "Fraction of wall time in GPU compute", lbl, registry=self.reg)


def create_app(engine: Engine, info: dict, name: str = "worker") -> FastAPI:
    app = FastAPI(title=f"llmserve worker {name}")
    tok = engine.lm.tokenizer
    m = _Metrics(name)
    state = {"ready": True}

    def encode(body: dict) -> list[int]:
        if "prompt_ids" in body:
            return list(body["prompt_ids"])
        if "messages" in body:
            return tok.apply_chat_template(body["messages"], add_generation_prompt=True)
        prompt = body.get("prompt")
        if prompt is None:
            raise HTTPException(400, "need prompt, prompt_ids or messages")
        if body.get("chat", True):
            return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        return tok.encode(prompt)

    @app.get("/health")
    async def health():
        alive = engine._thread is not None and engine._thread.is_alive()
        return JSONResponse({"status": "ok" if alive and state["ready"] else "down", "name": name},
                            status_code=200 if alive else 503)

    @app.get("/stats")
    async def stats():
        s = engine.stats()
        s.update({"name": name, "model": info.get("model_id"), "precision": info.get("precision"),
                  "weight_bytes": info.get("weight_bytes_tensors"), "device": info.get("device_name")})
        return s

    @app.get("/info")
    async def get_info():
        return info

    @app.get("/metrics")
    async def metrics():
        s = engine.stats()
        m.queue.labels(name).set(s["queue_depth"])
        m.active.labels(name).set(s["active"])
        m.kv_free.labels(name).set(s["kv_free_tokens"])
        m.mem.labels(name).set(s["mem_allocated_bytes"])
        m.busy.labels(name).set(s["busy_fraction"])
        return PlainTextResponse(generate_latest(m.reg).decode(), media_type="text/plain; version=0.0.4")

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        ids = encode(body)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        t_recv = time.perf_counter()

        def on_token(t: int, ts: float) -> None:
            loop.call_soon_threadsafe(q.put_nowait, ("tok", t))

        def on_finish(summary: dict) -> None:
            loop.call_soon_threadsafe(q.put_nowait, ("done", summary))

        req = GenRequest(
            prompt_ids=ids,
            max_new_tokens=int(body.get("max_tokens", 128)),
            temperature=float(body.get("temperature", 0.0)),
            ignore_eos=bool(body.get("ignore_eos", False)),
            seed=body.get("seed"),
            on_token=on_token,
            on_finish=on_finish,
        )
        if not engine.submit(req):
            if req.finish_reason == "too_long":
                raise HTTPException(413, "prompt + max_tokens exceeds worker KV budget")
            raise HTTPException(503, "worker queue full")

        def record(summary: dict) -> None:
            m.requests.labels(name, summary["finish_reason"]).inc()
            m.tokens.labels(name).inc(summary["completion_tokens"])
            if summary.get("ttft_s") is not None:
                m.ttft.labels(name).observe(summary["ttft_s"])
            if summary.get("e2e_s") is not None:
                m.e2e.labels(name).observe(summary["e2e_s"])

        if body.get("stream", False):

            async def sse():
                out_ids: list[int] = []
                sent = ""
                try:
                    while True:
                        kind, val = await q.get()
                        if kind == "tok":
                            out_ids.append(val)
                            text = tok.decode(out_ids, skip_special_tokens=True)
                            delta, sent = text[len(sent):], text
                            yield f"data: {json.dumps({'t': delta})}\n\n"
                        else:
                            record(val)
                            val.update({"done": True, "worker": name, "server_s": time.perf_counter() - t_recv})
                            yield f"data: {json.dumps(val)}\n\n"
                            return
                except asyncio.CancelledError:
                    req.cancelled = True
                    raise

            return StreamingResponse(sse(), media_type="text/event-stream")

        try:
            while True:
                kind, val = await q.get()
                if kind == "done":
                    break
        except asyncio.CancelledError:
            req.cancelled = True
            raise
        record(val)
        val.update({"worker": name, "text": tok.decode(req.output_ids, skip_special_tokens=True),
                    "server_s": time.perf_counter() - t_recv})
        return val

    @app.on_event("shutdown")
    async def _shutdown():
        engine.stop()

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="llmserve model worker")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "int8", "int4"])
    ap.add_argument("--quant-backend", default="native", choices=["native", "bnb"])
    ap.add_argument("--no-quantize-embeddings", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--policy", default="continuous", choices=["serial", "static", "continuous"])
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--window-ms", type=float, default=20.0)
    ap.add_argument("--kv-budget-tokens", type=int, default=65536)
    ap.add_argument("--max-prefill-tokens", type=int, default=4096)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    lm = load_model(a.model, a.precision, device=a.device, backend=a.quant_backend,
                    quantize_embeddings=not a.no_quantize_embeddings)
    cfg = EngineConfig(policy=a.policy, max_batch=a.max_batch, window_ms=a.window_ms,
                       kv_budget_tokens=a.kv_budget_tokens, max_prefill_tokens=a.max_prefill_tokens)
    engine = Engine(lm, cfg).start()
    # warm-up so the first measured request does not pay kernel compilation
    from .engine import generate_blocking

    generate_blocking(engine, lm.tokenizer.encode("Hello, warm up the kernels."), max_new_tokens=4)
    engine.started_at = time.perf_counter()
    engine.busy_s = 0.0
    name = a.name or f"{a.precision}-{a.policy}-{a.port}"
    info = dict(lm.info, engine=vars(cfg), name=name)
    app = create_app(engine, info, name)
    print(json.dumps({"event": "worker_ready", "name": name, "port": a.port}), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
