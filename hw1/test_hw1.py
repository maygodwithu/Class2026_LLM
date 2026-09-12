import torch

import tokenizer
from model import GPT, GPTConfig


def make_cfg(vocab_size=50, n_embd=16, block_size=32, dropout=0.0):
    return GPTConfig(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, dropout=dropout)


def test_output_shape():
    m = GPT(make_cfg())
    idx = torch.randint(0, 50, (4, 10))
    out = m(idx)
    assert out.shape == (4, 10, 16)


def test_matches_manual_lookup():
    m = GPT(make_cfg())
    idx = torch.tensor([[3, 7, 3]])
    out = m(idx)
    pos = torch.arange(3)
    expected = m.transformer.wte(idx) + m.transformer.wpe(pos)
    assert torch.allclose(out, expected, atol=1e-6)


def test_same_token_different_position_differs():
    m = GPT(make_cfg())
    idx = torch.tensor([[5, 5, 5]])
    out = m(idx)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_gradients_flow():
    m = GPT(make_cfg())
    idx = torch.randint(0, 50, (2, 5))
    out = m(idx)
    out.sum().backward()
    assert m.transformer.wte.weight.grad is not None
    assert m.transformer.wpe.weight.grad is not None


def test_uses_moduledict_named_transformer():
    m = GPT(make_cfg())
    assert isinstance(m.transformer, torch.nn.ModuleDict)
    assert "wte" in m.transformer and "wpe" in m.transformer and "drop" in m.transformer


# tokenizer.py 확인용 (구현할 것 없음, tokenizer.py 참고)
def test_tokenizer_splits_a_word_into_multiple_tokens():
    text = "The extraordinarily tiny robot whirred."
    ids = tokenizer.encode(text)
    print(f"\ntext: {text!r}\nids: {ids}\nwords: {len(text.split())}  tokens: {len(ids)}")

    assert tokenizer.decode(ids) == text
    assert len(ids) > len(text.split())


# text -> tokenizer.py -> model.py 전체 확인
def test_text_to_model_end_to_end():
    text = "Once upon a time, there was a small robot."
    ids = tokenizer.encode(text)
    idx = torch.tensor([ids])

    m = GPT(make_cfg(vocab_size=tokenizer.vocab_size, n_embd=32, block_size=64))
    out = m(idx)
    assert out.shape == (1, len(ids), 32)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
