"""Load a Hugging Face causal LM in a chosen numerical precision."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import device as dev
from .quant import quantize_model

# precision name -> (compute dtype, weight bits or None)
PRECISIONS: dict[str, tuple[torch.dtype, int | None]] = {
    "fp32": (torch.float32, None),
    "fp16": (torch.float16, None),
    "bf16": (torch.bfloat16, None),
    "int8": (torch.float16, 8),
    "int4": (torch.float16, 4),
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    precision: str
    compute_dtype: torch.dtype
    info: dict = field(default_factory=dict)

    @property
    def kv_bytes_per_token(self) -> int:
        c = self.model.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        kv_heads = getattr(c, "num_key_value_heads", c.num_attention_heads)
        elt = torch.finfo(self.compute_dtype).bits // 8
        return 2 * c.num_hidden_layers * kv_heads * head_dim * elt


def load_model(
    model_id: str,
    precision: str = "fp16",
    device: str | None = None,
    group_size: int = 64,
    quantize_embeddings: bool = True,
    backend: str = "native",
    fast_gqa: bool = True,
) -> LoadedModel:
    """Load ``model_id`` and convert it to ``precision``.

    ``backend="bnb"`` uses bitsandbytes (CUDA only) for int8/int4 instead of the
    built-in quantizer.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {list(PRECISIONS)}")
    if fast_gqa:
        from .attention import patch_qwen2_gqa

        patch_qwen2_gqa()
    d = dev.pick_device(device)
    dtype, bits = PRECISIONS[precision]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dev.empty_cache(d)
    mem0 = dev.memory_allocated(d)

    if bits is not None and backend == "bnb":
        from transformers import BitsAndBytesConfig

        qcfg = (
            BitsAndBytesConfig(load_in_8bit=True)
            if bits == 8
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qcfg, device_map={"": d.index or 0}, attn_implementation="sdpa"
        )
        qinfo = {"backend": "bitsandbytes"}
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa")
        model.to(d)
        qinfo = {}
        if bits is not None:
            rep = quantize_model(model, bits, group_size=group_size, quantize_embeddings=quantize_embeddings)
            qinfo = {"backend": "native", **asdict(rep)}
            dev.empty_cache(d)
    model.eval()
    dev.synchronize(d)
    load_s = time.perf_counter() - t0
    weight_bytes = dev.memory_allocated(d) - mem0

    n_params = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for n, b in model.named_buffers() if "qweight" in n
    )
    info = {
        "model_id": model_id,
        "precision": precision,
        "device": str(d),
        "device_name": dev.describe(d),
        "load_seconds": round(load_s, 2),
        "weight_bytes_measured": int(weight_bytes),
        "weight_bytes_tensors": int(_tensor_bytes(model)),
        "n_params_or_codes": int(n_params),
        "quant": qinfo,
        "fast_gqa": fast_gqa,
    }
    lm = LoadedModel(model=model, tokenizer=tok, device=d, precision=precision, compute_dtype=dtype, info=info)
    info["kv_bytes_per_token"] = lm.kv_bytes_per_token
    return lm


def _tensor_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def eos_ids(lm: LoadedModel) -> set[int]:
    ids: set[int] = set()
    gc = getattr(lm.model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if e is None:
        e = lm.tokenizer.eos_token_id
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(e)
    return ids
