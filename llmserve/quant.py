"""Weight-only post-training quantization for decoder LLMs.

Two formats are implemented from scratch so that they behave identically on
CUDA, Apple MPS and CPU:

* ``int8`` - symmetric, per-output-channel. ``W ~= q * s`` with ``q`` in
  [-127, 127] stored as int8 and one fp16 scale per output row.
* ``int4`` - asymmetric, group-wise along the input dimension (default group
  of 64 columns). ``W ~= (q - z) * s`` with ``q`` in [0, 15]; two codes are
  packed into one uint8, and every group has its own fp16 scale and zero point.
  This is the same storage layout used by GPTQ/AWQ-style 4-bit checkpoints,
  but here the codes come from plain round-to-nearest (RTN) quantization.

Activations stay in the compute dtype (fp16/bf16). At run time the weight is
dequantized into the compute dtype right before the matmul. This saves resident
memory (weights are 2x / ~3.6x smaller) but adds a dequantization pass to every
forward call, which is exactly the trade-off measured in the report. Fused
low-bit GEMM kernels (Marlin, AWQ, bitsandbytes) remove most of that overhead on
CUDA; ``backend="bnb"`` in :func:`quantize_model` uses bitsandbytes when it is
installed.

Tied input embeddings and LM head share one quantized tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Core quantize / dequantize routines
# --------------------------------------------------------------------------- #


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8. Returns (int8 codes [out,in], fp16 scales [out])."""
    w32 = w.float()
    amax = w32.abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.round(w32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)[:, None]


def quantize_int4(
    w: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric group-wise 4-bit.

    Returns ``(packed uint8 [out, in/2], scales fp16 [out, in/g], zeros fp16 [out, in/g])``.
    """
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    wmin = g.amin(dim=2)
    wmax = g.amax(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    zero = torch.round(-wmin / scale).clamp(0, 15)
    q = torch.round(g / scale[..., None] + zero[..., None]).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_f, in_f)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.to(torch.float16), zero.to(torch.float16)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    # NB: uint8 ``>>`` returns wrong results on MPS in PyTorch 2.5, so use
    # integer division, which is correct on every backend.
    lo = packed & 0x0F
    hi = torch.div(packed, 16, rounding_mode="floor")
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def dequantize_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    q = unpack_int4(packed).to(dtype)
    out_f, in_f = q.shape
    q = q.reshape(out_f, in_f // group_size, group_size)
    w = (q - zero.to(dtype)[..., None]) * scale.to(dtype)[..., None]
    return w.reshape(out_f, in_f)


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #


class QuantLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding int8 or int4 weights."""

    def __init__(self, linear: nn.Linear, bits: int, group_size: int, compute_dtype: torch.dtype):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        w = linear.weight.data
        if bits == 8:
            q, s = quantize_int8(w)
            self.register_buffer("qweight", q)
            self.register_buffer("scales", s)
            self.zeros = None
        elif bits == 4:
            q, s, z = quantize_int4(w, group_size)
            self.register_buffer("qweight", q)
            self.register_buffer("scales", s)
            self.register_buffer("zeros", z)
        else:
            raise ValueError(f"unsupported bits={bits}")
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(compute_dtype))
        else:
            self.bias = None

    def dequantize(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_int8(self.qweight, self.scales, self.compute_dtype)
        return dequantize_int4(self.qweight, self.scales, self.zeros, self.group_size, self.compute_dtype)

    def dequantize_rows(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantize only the requested output rows (used for embedding lookups)."""
        if self.bits == 8:
            return self.qweight[idx].to(self.compute_dtype) * self.scales[idx].to(self.compute_dtype)[..., None]
        packed = self.qweight[idx]
        flat = packed.reshape(-1, packed.shape[-1])
        q = unpack_int4(flat).to(self.compute_dtype)
        q = q.reshape(flat.shape[0], -1, self.group_size)
        s = self.scales[idx].reshape(flat.shape[0], -1).to(self.compute_dtype)
        z = self.zeros[idx].reshape(flat.shape[0], -1).to(self.compute_dtype)
        w = (q - z[..., None]) * s[..., None]
        return w.reshape(*idx.shape, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the weight *before* the matmul: folding the scale into the output
        # instead overflows fp16, because unscaled int8 codes reach +-127.
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, group={self.group_size}"


class QuantEmbedding(nn.Module):
    """Embedding lookup that reads rows out of a (possibly shared) QuantLinear."""

    def __init__(self, source: QuantLinear, padding_idx: int | None = None):
        super().__init__()
        self.source = source
        self.padding_idx = padding_idx
        self.embedding_dim = source.in_features
        self.num_embeddings = source.out_features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.source.dequantize_rows(input_ids)


# --------------------------------------------------------------------------- #
# Model surgery
# --------------------------------------------------------------------------- #


@dataclass
class QuantReport:
    bits: int
    group_size: int
    n_linear: int
    params_quantized: int
    params_total: int
    bytes_before: int
    bytes_after: int


def _module_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def quantize_model(
    model: nn.Module,
    bits: int,
    group_size: int = 64,
    quantize_embeddings: bool = True,
    skip: tuple[str, ...] = (),
) -> QuantReport:
    """Replace every ``nn.Linear`` (and optionally the tied embedding/LM head)
    with a quantized equivalent, in place. The model should already be on its
    target device in its compute dtype."""
    compute_dtype = next(model.parameters()).dtype
    bytes_before = _module_bytes(model)
    params_total = sum(p.numel() for p in model.parameters())

    lm_head = getattr(model, "lm_head", None)
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tied = (
        lm_head is not None
        and embed is not None
        and lm_head.weight.data_ptr() == embed.weight.data_ptr()
    )

    n = 0
    pq = 0
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        if mod is lm_head and not quantize_embeddings:
            continue
        targets.append((name, mod))

    for name, mod in targets:
        qmod = QuantLinear(mod, bits, group_size, compute_dtype)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, qmod)
        n += 1
        pq += mod.weight.numel()
        if mod is lm_head and tied:
            emb_parent_name, _, emb_child = _find_module_name(model, embed).rpartition(".")
            emb_parent = model.get_submodule(emb_parent_name) if emb_parent_name else model
            setattr(emb_parent, emb_child, QuantEmbedding(qmod, getattr(embed, "padding_idx", None)))
        del mod

    if quantize_embeddings and embed is not None and not tied:
        src = nn.Linear(embed.embedding_dim, embed.num_embeddings, bias=False)
        src.weight = embed.weight
        qsrc = QuantLinear(src, bits, group_size, compute_dtype)
        name = _find_module_name(model, embed)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, QuantEmbedding(qsrc, getattr(embed, "padding_idx", None)))
        pq += embed.weight.numel()

    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False  # stop HF from re-tying weights

    return QuantReport(
        bits=bits,
        group_size=group_size,
        n_linear=n,
        params_quantized=pq,
        params_total=params_total,
        bytes_before=bytes_before,
        bytes_after=_module_bytes(model),
    )


def _find_module_name(model: nn.Module, target: nn.Module) -> str:
    for name, mod in model.named_modules():
        if mod is target:
            return name
    raise KeyError("module not found")
