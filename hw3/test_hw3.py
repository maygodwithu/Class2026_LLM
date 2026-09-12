import torch

from block import Block, MLP


def test_mlp_shape_and_expansion():
    m = MLP(n_embd=16, dropout=0.0)
    x = torch.randn(2, 5, 16)
    out = m(x)
    assert out.shape == (2, 5, 16)
    assert m.c_fc.out_features == 64, "hidden layer should be a 4x expansion (16 -> 64)"


def test_mlp_gradients_flow():
    m = MLP(n_embd=16, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = m(x)
    out.sum().backward()
    assert x.grad is not None
    assert m.c_fc.weight.grad is not None


def test_block_shape_preserved():
    b = Block(n_embd=16, n_head=4, dropout=0.0)
    x = torch.randn(2, 5, 16)
    out = b(x)
    assert out.shape == x.shape


def test_block_is_residual():
    """With the sublayers' output projections zeroed out, attn(x) and
    mlp(x) both output all zeros -- so if the residual wiring is correct,
    Block(x) must equal x exactly. If this fails, x is probably being
    overwritten instead of added to."""
    b = Block(n_embd=16, n_head=4, dropout=0.0)
    with torch.no_grad():
        for p in b.attn.c_proj.parameters():
            p.zero_()
        for p in b.mlp.c_proj.parameters():
            p.zero_()
    x = torch.randn(2, 5, 16)
    out = b(x)
    assert torch.allclose(out, x, atol=1e-4), \
        "with zeroed sublayer outputs, Block(x) should equal x -- check the residual (x = x + ...) wiring"


def test_block_gradients_flow():
    b = Block(n_embd=16, n_head=4, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = b(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
