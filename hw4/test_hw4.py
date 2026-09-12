import math

import torch
import torch.nn as nn

from model import GPT, GPTConfig


def make_cfg(vocab_size=37):
    return GPTConfig(vocab_size=vocab_size, n_embd=16, n_layer=2, n_head=4, block_size=8, dropout=0.0)


def _small_init(model: nn.Module):
    """Force small, well-behaved weights regardless of whatever default
    init GPT.__init__ happens to use -- this course only requires you to
    WIRE the modules together correctly, not to design an init scheme.
    (Weight tying means wte and lm_head share one tensor, so this also
    only needs to touch it once.)"""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)


def test_weight_tying():
    cfg = make_cfg()
    m = GPT(cfg)
    assert m.lm_head.weight.data_ptr() == m.wte.weight.data_ptr(), \
        "lm_head.weight should be the SAME tensor object as wte.weight (weight tying), not just equal values"


def test_forward_shapes_no_targets():
    cfg = make_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 6))
    logits, loss = m(idx)
    assert logits.shape == (3, 6, cfg.vocab_size)
    assert loss is None


def test_forward_loss_with_targets():
    cfg = make_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 6))
    targets = torch.randint(0, cfg.vocab_size, (3, 6))
    logits, loss = m(idx, targets)
    assert logits.shape == (3, 6, cfg.vocab_size)
    assert loss is not None and loss.dim() == 0


def test_loss_near_random_init_baseline():
    """An untrained model should have loss close to ln(vocab_size) -- i.e.
    it's basically guessing uniformly at random over the vocabulary. This
    is the exact sanity check the course's reference train.py logs at
    step 0 of every training run (and it matched ln(vocab_size) closely
    in practice when we ran it)."""
    torch.manual_seed(0)
    cfg = make_cfg(vocab_size=100)
    m = GPT(cfg)
    _small_init(m)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    _, loss = m(idx, targets)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, \
        f"loss {loss.item():.3f} too far from ln(vocab_size)={expected:.3f} for an untrained model"


def test_generate_length_and_block_size_cropping():
    cfg = make_cfg()
    m = GPT(cfg)
    # prompt already longer than block_size -- generate() must crop internally
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 3))
    out = m.generate(idx, max_new_tokens=5, top_k=5)
    assert out.shape == (1, cfg.block_size + 3 + 5)


def test_generate_respects_top_k():
    """With top_k=1, generation is deterministic (always the argmax)."""
    torch.manual_seed(0)
    cfg = make_cfg()
    m = GPT(cfg)
    m.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 3))
    out1 = m.generate(idx.clone(), max_new_tokens=4, top_k=1)
    out2 = m.generate(idx.clone(), max_new_tokens=4, top_k=1)
    assert torch.equal(out1, out2), "top_k=1 should be deterministic"


def test_gradients_flow_through_full_model():
    cfg = make_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = m(idx, targets)
    loss.backward()
    assert m.wte.weight.grad is not None
    assert m.blocks[0].attn.c_attn.weight.grad is not None
    assert m.blocks[-1].mlp.c_fc.weight.grad is not None


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
