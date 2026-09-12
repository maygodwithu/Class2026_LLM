"""
HW2 -- Causal Self-Attention

Goal: implement multi-head causal self-attention -- the mechanism that
lets each position in a sequence gather information from a weighted
combination of all EARLIER positions, but never the future.

TODO: implement CausalSelfAttention.forward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # One fused linear for q, k, v instead of three separate ones --
        # same math, one matmul instead of three.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) where C == n_embd

        Steps:
          1. Project x to q, k, v in one shot via self.c_attn(x), which
             gives you a (B, T, 3*C) tensor -- split it into three
             (B, T, C) pieces with .split(self.n_embd, dim=2).
          2. Reshape each of q, k, v from (B, T, C) to (B, n_head, T, head_dim)
             so attention is computed independently per head:
             .view(B, T, n_head, head_dim).transpose(1, 2)
             (head_dim = C // n_head)
          3. Compute causal scaled dot-product attention with
             F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=True).
             is_causal=True is what masks out the future -- position i can
             only attend to positions <= i.
          4. Reshape the result back from (B, n_head, T, head_dim) to
             (B, T, C): .transpose(1, 2).contiguous().view(B, T, C)
          5. Apply the output projection: self.resid_dropout(self.c_proj(y))

        Returns: (B, T, C)
        """
        B, T, C = x.shape
        # TODO: implement
        raise NotImplementedError
