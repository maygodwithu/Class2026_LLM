"""
A small, readable GPT-style decoder-only transformer (nanoGPT-style).

Architecture: token + learned positional embeddings -> N pre-norm
transformer blocks (causal self-attention + MLP) -> final LayerNorm ->
linear head tied to the token embedding.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    block_size: int
    dropout: float = 0.1
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    PyTorch's built-in nn.LayerNorm doesn't let bias=False, but skipping
    the bias term is a common small-model trick (fewer params, slightly
    more stable training) -- worth exposing explicitly rather than hiding
    it inside a library call.
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # One fused linear for q, k, v instead of three separate ones --
        # same math, one matmul instead of three.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # is_causal=True masks each position to only attend to itself and
        # earlier positions -- this is what makes it a *language* model
        # (can't see the future) rather than a bidirectional encoder.
        # scaled_dot_product_attention picks a fused kernel under the hood
        # (flash-attention-style on supported GPUs) instead of materializing
        # a full (T, T) attention matrix.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-norm + residual: normalize *before* each sublayer, and add
        # the sublayer's output back to the (un-normalized) input. This is
        # what makes very deep transformers trainable -- gradients have a
        # clean residual path back to the input regardless of depth.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the output head reuses the input embedding matrix
        # instead of learning a second (vocab_size x n_embd) matrix. At
        # this tiny model scale the vocab embedding is often 20-35% of all
        # parameters (see configs.py) -- tying effectively cuts that cost
        # in half, and empirically doesn't hurt (and often helps) quality.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual projections, to keep the
        # variance of activations roughly constant regardless of depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding_position_only: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding_position_only:
            # wpe (positional embedding) isn't tied/shared with anything,
            # but nanoGPT convention excludes it from the "headline" count
            # since it doesn't scale with vocab_size. Included in the raw
            # total either way -- this flag just changes what's reported.
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
