import torch

from llmserve.quant import (QuantLinear, dequantize_int4, dequantize_int8, quantize_int4, quantize_int8,
                            quantize_model, unpack_int4)


def test_int8_roundtrip_error_bounded():
    w = torch.randn(64, 128)
    q, s = quantize_int8(w)
    err = (dequantize_int8(q, s, torch.float32) - w).abs()
    assert q.dtype == torch.int8
    assert (err <= s.float()[:, None] / 2 + 1e-3).all()  # at most half a quantization step


def test_int4_pack_unpack_and_error():
    w = torch.randn(32, 128)
    packed, s, z = quantize_int4(w, 64)
    assert packed.shape == (32, 64) and packed.dtype == torch.uint8
    codes = unpack_int4(packed)
    assert codes.max() <= 15 and codes.min() >= 0
    err = (dequantize_int4(packed, s, z, 64, torch.float32) - w).abs()
    assert err.max() < (s.float().max() * 1.01)


def test_quantlinear_close_to_linear():
    lin = torch.nn.Linear(128, 64)
    x = torch.randn(3, 128)
    for bits, tol in [(8, 0.05), (4, 0.6)]:
        ql = QuantLinear(lin, bits, 64, torch.float32)
        assert (ql(x) - lin(x)).abs().max() < tol


def test_quantize_model_shrinks_and_keeps_tied_embedding(tiny_lm):
    import copy

    from llmserve.quant import QuantEmbedding

    m = copy.deepcopy(tiny_lm.model)
    rep = quantize_model(m, bits=8, group_size=64)
    assert rep.bytes_after < rep.bytes_before / 2.5
    assert isinstance(m.model.embed_tokens, QuantEmbedding)
    assert m.model.embed_tokens.source is m.lm_head  # one shared quantized tensor
    out = m(input_ids=torch.tensor([[5, 6, 7]])).logits
    assert torch.isfinite(out).all()
