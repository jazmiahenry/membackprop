"""Parity of the from-scratch Llama with Hugging Face transformers."""

import pytest
import torch
from transformers import LlamaConfig as HFConfig
from transformers import LlamaForCausalLM

from membackprop.llama import Llama, LlamaConfig

SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
    "rope_theta": 10000.0,
}
PLAIN = {"rope_type": "default", "rope_theta": 10000.0}


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("rope", [PLAIN, SCALING], ids=["plain", "llama3"])
def test_matches_transformers(rope, tied):
    """Logits and gradients agree with the reference implementation."""
    torch.manual_seed(0)
    hf_cfg = HFConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
        rope_parameters=rope,
    )
    reference = LlamaForCausalLM(hf_cfg).eval()
    model = Llama(LlamaConfig.from_hf(hf_cfg.to_dict()))
    model.load_hf_state_dict(reference.state_dict())
    tokens = torch.randint(0, 97, (2, 100))
    expected, actual = reference(tokens).logits, model.lm_head(model(tokens))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    expected.pow(2).mean().backward()
    actual.pow(2).mean().backward()
    torch.testing.assert_close(
        model.layers[0].self_attn.k_proj.weight.grad,
        reference.model.layers[0].self_attn.k_proj.weight.grad,
        rtol=1e-3,
        atol=1e-6,
    )
