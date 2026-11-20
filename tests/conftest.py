import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


@pytest.fixture(scope="session")
def tiny_lm():
    """A randomly initialised 2-layer Qwen2 (GQA 4 query / 2 KV heads) on CPU, fp32."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from llmserve.attention import patch_qwen2_gqa
    from llmserve.model_loader import LoadedModel

    patch_qwen2_gqa()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = 1
    return LoadedModel(model=model, tokenizer=_Tok(), device=torch.device("cpu"), precision="fp32",
                       compute_dtype=torch.float32)
