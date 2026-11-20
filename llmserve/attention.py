"""GQA-aware SDPA attention for Qwen2 models (patch for transformers 4.46).

Grouped-query attention (GQA) shares each key/value head among ``G`` query
heads (Qwen2.5-0.5B: 14 query heads, 2 KV heads, G = 7). The stock Hugging Face
implementation calls ``repeat_kv``, which *materializes* a G-times larger copy of
the whole KV cache in every layer on every decode step. On a decode step this
copy costs far more than the attention itself: for batch 16 x 1024 cached
tokens we measured 3.4 ms per K or V copy versus 1.5 ms for the SDPA kernel, i.e.
~200 ms of a 230 ms step across 24 layers.

This patch folds the G query heads that share a KV head into the query-length
axis instead (``[B, Hkv, G*L, D]``), so SDPA reads the un-expanded KV cache
directly. The math is identical; only memory traffic changes. The causal
/padding mask is broadcast to the folded layout. When HF passes no mask and
``q_len > 1`` (unpadded prefill relying on ``is_causal``) we fall back to the
original path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_PATCHED = False


def _gqa_sdpa_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if position_embeddings is None:
        cos, sin = self.rotary_emb(v, position_ids)
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    G = self.num_key_value_groups
    T = k.shape[-2]
    mask = attention_mask[:, :, :, :T] if attention_mask is not None else None

    if G == 1 or (mask is None and q_len > 1):
        k, v = repeat_kv(k, G), repeat_kv(v, G)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and q_len > 1)
    else:
        Hk = self.num_key_value_heads
        qf = q.reshape(bsz, Hk, G * q_len, self.head_dim)  # heads are ordered kv_head*G + g
        if mask is not None:
            mb = mask.shape[0]
            mask = mask.unsqueeze(2).expand(mb, 1, G, q_len, T).reshape(mb, 1, G * q_len, T)
        out = F.scaled_dot_product_attention(qf, k, v, attn_mask=mask)
        out = out.reshape(bsz, self.num_heads, q_len, self.head_dim)

    out = out.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    return self.o_proj(out), None, past_key_value


def patch_qwen2_gqa() -> None:
    """Install the GQA-aware forward on ``Qwen2SdpaAttention`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.Qwen2SdpaAttention.forward = _gqa_sdpa_forward
    _PATCHED = True
