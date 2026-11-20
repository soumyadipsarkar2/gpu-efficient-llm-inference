import threading
import time

import pytest
import torch

from llmserve.engine import Engine, EngineConfig, GenRequest


def reference_greedy(lm, prompt, n):
    ids = torch.tensor([prompt])
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = lm.model(input_ids=ids).logits[0, -1]
            t = int(logits.argmax())
            out.append(t)
            ids = torch.cat([ids, torch.tensor([[t]])], 1)
    return out


PROMPTS = [[5, 9, 22, 7], [3] * 17, [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], [8, 2], list(range(10, 40))]


@pytest.mark.parametrize("policy", ["serial", "static", "continuous"])
def test_batched_outputs_match_unbatched(tiny_lm, policy):
    refs = [reference_greedy(tiny_lm, p, 12) for p in PROMPTS]
    eng = Engine(tiny_lm, EngineConfig(policy=policy, max_batch=4, window_ms=5)).start()
    reqs, evs = [], []
    for i, p in enumerate(PROMPTS):  # staggered arrivals force joins into a running batch
        ev = threading.Event()
        r = GenRequest(prompt_ids=p, max_new_tokens=12, ignore_eos=True, on_finish=lambda s, ev=ev: ev.set())
        eng.submit(r)
        reqs.append(r)
        evs.append(ev)
        time.sleep(0.01 * i)
    for ev in evs:
        assert ev.wait(30)
    eng.stop()
    for r, ref in zip(reqs, refs):
        assert r.output_ids == ref
    st = eng.stats()
    assert st["finished"] == len(PROMPTS) and st["active"] == 0
    if policy == "continuous":
        assert max(st["batch_hist"]) > 1


def test_kv_budget_limits_admission(tiny_lm):
    eng = Engine(tiny_lm, EngineConfig(policy="continuous", max_batch=8, kv_budget_tokens=40)).start()
    evs = []
    for _ in range(4):
        ev = threading.Event()
        eng.submit(GenRequest(prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=15, ignore_eos=True,
                              on_finish=lambda s, ev=ev: ev.set()))
        evs.append(ev)
    for ev in evs:
        assert ev.wait(30)
    eng.stop()
    assert max(eng.stats()["batch_hist"]) <= 2  # 20 tokens reserved each, budget 40


def test_oversized_request_rejected(tiny_lm):
    eng = Engine(tiny_lm, EngineConfig(kv_budget_tokens=10)).start()
    got = {}
    ok = eng.submit(GenRequest(prompt_ids=[1] * 8, max_new_tokens=8, on_finish=lambda s: got.update(s)))
    eng.stop()
    assert not ok and got["finish_reason"] == "too_long"


def test_gqa_patch_matches_reference_attention(tiny_lm):
    from transformers import DynamicCache

    x = torch.tensor([[5, 6, 7, 8, 9, 10], [11, 12, 13, 0, 0, 0]])
    m = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        patched = tiny_lm.model(input_ids=x, attention_mask=m).logits
        from transformers.models.qwen2 import modeling_qwen2

        cur = modeling_qwen2.Qwen2SdpaAttention.forward
        modeling_qwen2.Qwen2SdpaAttention.forward = modeling_qwen2.Qwen2SdpaAttention.__mro__[1].forward
        try:
            ref = tiny_lm.model(input_ids=x, attention_mask=m).logits
        finally:
            modeling_qwen2.Qwen2SdpaAttention.forward = cur
    assert torch.allclose(patched[0], ref[0], atol=1e-5)
    assert torch.allclose(patched[1, :3], ref[1, :3], atol=1e-5)
