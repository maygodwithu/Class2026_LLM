import torch

from attention import CausalSelfAttention


def test_output_shape():
    m = CausalSelfAttention(n_embd=16, n_head=4, dropout=0.0)
    x = torch.randn(2, 5, 16)
    out = m(x)
    assert out.shape == (2, 5, 16)


def test_causal_masking():
    """Changing a FUTURE token must not change an EARLIER position's output.
    This is the defining property of causal (as opposed to bidirectional)
    attention -- if this fails, information is leaking from the future."""
    torch.manual_seed(0)
    m = CausalSelfAttention(n_embd=16, n_head=4, dropout=0.0)
    m.eval()
    x = torch.randn(1, 6, 16)

    out1 = m(x)
    x2 = x.clone()
    x2[0, 5] = torch.randn(16)  # perturb ONLY the last position

    out2 = m(x2)

    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5), \
        "output at earlier positions changed when only a LATER token changed -- attention is not causal"
    assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-5), \
        "position 5's own output should change when its own input changes"


def test_multihead_reshape_consistency():
    """head_dim must be n_embd // n_head, and different n_head choices
    should still produce the right output shape."""
    for n_head in (1, 2, 4, 8):
        m = CausalSelfAttention(n_embd=16, n_head=n_head, dropout=0.0)
        x = torch.randn(3, 4, 16)
        out = m(x)
        assert out.shape == (3, 4, 16), f"failed for n_head={n_head}"


def test_gradients_flow():
    m = CausalSelfAttention(n_embd=16, n_head=4, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = m(x)
    out.sum().backward()
    assert x.grad is not None
    assert m.c_attn.weight.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
