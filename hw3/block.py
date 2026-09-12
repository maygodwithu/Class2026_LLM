"""
HW3 -- Feed-Forward Network + Transformer Block

Goal: implement the MLP (feed-forward) sublayer, then assemble it with
the already-solved self-attention sublayer into one pre-norm residual
Transformer block -- the repeating unit that gets stacked N times to
build the full model in HW4.

LayerNorm and CausalSelfAttention below are given as working solutions
(from HW2) so this assignment can focus on the MLP and on how the two
sublayers combine.

TODO: implement MLP.forward() and Block.forward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Given -- LayerNorm with an optional bias."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Given -- solved in HW2."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)

        Apply: Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> dropout.

        Intuition: attention moves information BETWEEN positions; the MLP
        transforms information AT each position independently (every
        position goes through the exact same Linear-GELU-Linear, with no
        mixing across T). This is where most of the model's per-token
        "thinking" capacity lives -- in most GPT-style models the MLP
        holds roughly 2/3 of the non-embedding parameters.

        Returns: (B, T, C)
        """
        # TODO: implement
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)

        Pre-norm residual wiring:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

        Note the pattern: normalize BEFORE each sublayer, and ADD the
        sublayer's output back onto the (un-normalized) x -- don't
        overwrite x with the sublayer's output. This residual path is
        what keeps deep transformers trainable: gradients have a direct
        route back to the input regardless of how many blocks are stacked.

        Returns: (B, T, C)
        """
        # TODO: implement
        raise NotImplementedError
