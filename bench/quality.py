"""Model-quality evaluation across precisions (Experiment E2).

Three automated, reproducible metrics:

1. **Perplexity** on WikiText-2 (raw, test split): non-overlapping 1024-token
   windows, the first ``--ppl-windows`` of them.
2. **ARC-Easy accuracy** (test split, zero-shot, log-likelihood scoring of each
   answer option given ``Question: ...\\nAnswer:``). Reports ``acc`` (sum of
   log-probs) and ``acc_norm`` (log-prob per answer character).
3. **Greedy-output agreement** with the fp32 model on chat prompts drawn from the
   benchmark workload: exact-match rate over 64 generated tokens and mean length
   of the common prefix before the first diverging token. Generation goes through
   the serving engine (continuous batching), so this also checks the serving path.

Usage:
    python -m bench.quality --precision int4 --out results/quality/qwen0.5b_int4.json
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from llmserve.engine import Engine, EngineConfig, GenRequest
from llmserve.model_loader import load_model


@torch.no_grad()
def perplexity(lm, windows: int = 40, seqlen: int = 1024, batch: int = 4) -> float:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = lm.tokenizer(text, return_tensors="pt").input_ids[0]
    n = min(windows, ids.numel() // seqlen)
    chunks = ids[: n * seqlen].view(n, seqlen)
    nll, count = 0.0, 0
    for i in range(0, n, batch):
        x = chunks[i : i + batch].to(lm.device)
        logits = lm.model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
        count += x[:, 1:].numel()
    return math.exp(nll / count)


@torch.no_grad()
def arc_easy(lm, limit: int | None = None) -> dict:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    tok = lm.tokenizer
    correct = correct_norm = 0
    for ex in ds:
        ctx = f"Question: {ex['question']}\nAnswer:"
        ctx_ids = tok.encode(ctx)
        conts = [" " + c for c in ex["choices"]["text"]]
        seqs = [ctx_ids + tok.encode(c) for c in conts]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            m[i, : len(s)] = 1
        logp = torch.log_softmax(
            lm.model(input_ids=x.to(lm.device), attention_mask=m.to(lm.device), use_cache=False).logits.float(), -1
        ).cpu()
        scores, norms = [], []
        for i, s in enumerate(seqs):
            tgt = torch.tensor(s[len(ctx_ids):])
            lp = logp[i, len(ctx_ids) - 1 : len(s) - 1].gather(1, tgt[:, None]).sum().item()
            scores.append(lp)
            norms.append(lp / max(len(conts[i]), 1))
        gold = ex["choices"]["label"].index(ex["answerKey"])
        correct += int(max(range(len(scores)), key=scores.__getitem__) == gold)
        correct_norm += int(max(range(len(norms)), key=norms.__getitem__) == gold)
    return {"n": len(ds), "acc": correct / len(ds), "acc_norm": correct_norm / len(ds)}


def greedy_outputs(lm, prompts: list[str], max_new: int = 64) -> list[list[int]]:
    eng = Engine(lm, EngineConfig(policy="continuous", max_batch=16)).start()
    reqs, evs = [], []
    for p in prompts:
        ids = lm.tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        ev = threading.Event()
        r = GenRequest(prompt_ids=ids, max_new_tokens=max_new, ignore_eos=False,
                       on_finish=lambda s, ev=ev: ev.set())
        eng.submit(r)
        reqs.append(r)
        evs.append(ev)
    for ev in evs:
        ev.wait()
    eng.stop()
    return [r.output_ids for r in reqs]


def agreement(ref: list[list[int]], out: list[list[int]]) -> dict:
    exact = sum(a == b for a, b in zip(ref, out)) / len(ref)
    prefixes = []
    for a, b in zip(ref, out):
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        prefixes.append(k / max(len(a), 1))
    return {"exact_match": exact, "mean_prefix_frac": sum(prefixes) / len(prefixes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--no-quantize-embeddings", action="store_true")
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--arc-limit", type=int, default=None)
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--ref", default=None, help="JSON from the fp32 run, to compute agreement")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from bench.workload import generate

    prompts = [r.prompt for r in generate(a.n_prompts, "heavy_tail", seed=123)]
    prompts = [p for p in prompts if len(p) < 4000][: a.n_prompts]
    lm = load_model(a.model, a.precision, quantize_embeddings=not a.no_quantize_embeddings)
    t0 = time.time()
    res = {"model": a.model, "precision": a.precision, "info": lm.info}
    res["ppl_wikitext2"] = perplexity(lm, a.ppl_windows)
    print("ppl", res["ppl_wikitext2"], flush=True)
    res["arc_easy"] = arc_easy(lm, a.arc_limit)
    print("arc", res["arc_easy"], flush=True)
    outs = greedy_outputs(lm, prompts)
    res["greedy_outputs"] = outs
    if a.ref:
        ref = json.loads(Path(a.ref).read_text())["greedy_outputs"]
        res["agreement_vs_ref"] = agreement(ref, outs)
        print("agreement", res["agreement_vs_ref"], flush=True)
    res["sample_text"] = lm.tokenizer.decode(outs[0], skip_special_tokens=True)
    res["eval_seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
