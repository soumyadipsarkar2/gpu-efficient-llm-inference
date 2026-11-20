"""Reproducible workload generation.

A workload is a list of requests with an arrival offset, a prompt, and an output
length. Prompts are built from WikiText-2 (train split) paragraphs wrapped in a
task instruction and trimmed to a target token length with the model tokenizer,
so the *same* trace can be replayed against every serving configuration.

Request classes (prompt tokens / output tokens):
    short   32-128   /  32-64
    medium  256-512  /  64-128
    long    1024-1536/ 128-256

Arrival processes:
    closed   N concurrent users, each sends its next request when the previous
             one finishes (arrival offsets are ignored by the client)
    poisson  open loop, exponential inter-arrival times at ``rate`` req/s
    bursty   open loop, Markov-modulated: alternates ON periods at ``rate*burst``
             and OFF periods at ``rate/burst``, same long-run mean rate

Usage:
    python -m bench.workload --mix balanced --n 200 --arrival poisson --rate 4 --out traces/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

CLASSES = {
    "short": ((32, 128), (32, 64)),
    "medium": ((256, 512), (64, 128)),
    "long": ((1024, 1536), (128, 256)),
}

MIXES = {
    "short": {"short": 1.0},
    "medium": {"medium": 1.0},
    "long": {"long": 1.0},
    "balanced": {"short": 0.5, "medium": 0.35, "long": 0.15},
    "heavy_tail": {"short": 0.8, "long": 0.2},
}

TASKS = [
    "Summarize the following passage in a few sentences.",
    "List the key facts mentioned in the following text.",
    "Write three questions a student could answer after reading this passage.",
    "Explain the main topic of the following text to a high-school student.",
    "Rewrite the following passage in plain, simple English.",
]


@dataclass
class TraceRequest:
    idx: int
    t: float  # arrival offset in seconds
    cls: str
    prompt: str
    prompt_tokens: int  # chat-templated length
    max_tokens: int


@lru_cache(maxsize=1)
def _corpus() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    paras, cur = [], []
    for line in ds["text"]:
        line = line.strip()
        if not line or line.startswith("="):
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(line)
    return [p for p in paras if len(p) > 400]


@lru_cache(maxsize=2)
def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def make_prompt(rng: random.Random, target_tokens: int, tokenizer_name: str) -> tuple[str, int]:
    tok = _tokenizer(tokenizer_name)
    corpus = _corpus()
    task = rng.choice(TASKS)
    text_ids: list[int] = []
    while len(text_ids) < target_tokens:
        text_ids += tok.encode(" " + rng.choice(corpus))
    overhead = len(tok.apply_chat_template([{"role": "user", "content": task + "\n\n"}], add_generation_prompt=True))
    body = tok.decode(text_ids[: max(8, target_tokens - overhead)])
    prompt = f"{task}\n\n{body}"
    n = len(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True))
    return prompt, n


def arrivals(rng: random.Random, n: int, kind: str, rate: float, burst: float = 4.0,
             mean_on_s: float = 5.0) -> list[float]:
    if kind == "closed":
        return [0.0] * n
    if kind == "poisson":
        t, out = 0.0, []
        for _ in range(n):
            t += rng.expovariate(rate)
            out.append(t)
        return out
    if kind == "bursty":
        # two-state MMPP with equal mean dwell time in each state
        hi, lo = rate * 2 * burst / (burst + 1), rate * 2 / (burst + 1)
        t, out, on = 0.0, [], True
        switch_at = rng.expovariate(1 / mean_on_s)
        while len(out) < n:
            r = hi if on else lo
            dt = rng.expovariate(r)
            if t + dt > switch_at:
                t = switch_at
                on = not on
                switch_at = t + rng.expovariate(1 / mean_on_s)
                continue
            t += dt
            out.append(t)
        return out
    raise ValueError(kind)


def generate(n: int, mix: str, arrival: str = "closed", rate: float = 1.0, seed: int = 0,
             tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct", burst: float = 4.0,
             fixed_max_tokens: int | None = None) -> list[TraceRequest]:
    rng = random.Random(seed)
    classes, weights = zip(*MIXES[mix].items())
    ts = arrivals(rng, n, arrival, rate, burst)
    out = []
    for i in range(n):
        cls = rng.choices(classes, weights)[0]
        (plo, phi), (olo, ohi) = CLASSES[cls]
        prompt, ptoks = make_prompt(rng, rng.randint(plo, phi), tokenizer)
        mt = fixed_max_tokens or rng.randint(olo, ohi)
        out.append(TraceRequest(i, ts[i], cls, prompt, ptoks, mt))
    return out


def save(trace: list[TraceRequest], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in trace:
            f.write(json.dumps(asdict(r)) + "\n")


def load(path: str | Path) -> list[TraceRequest]:
    with Path(path).open() as f:
        return [TraceRequest(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mix", default="balanced", choices=list(MIXES))
    ap.add_argument("--arrival", default="closed", choices=["closed", "poisson", "bursty"])
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--burst", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = generate(a.n, a.mix, a.arrival, a.rate, a.seed, burst=a.burst, fixed_max_tokens=a.max_tokens)
    save(tr, a.out)
    print(f"wrote {len(tr)} requests to {a.out}")


if __name__ == "__main__":
    main()
