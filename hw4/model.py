"""
HW4 -- Full Model Assembly + Decoding

Goal: stack Transformer blocks into a complete GPT, wire up the output
head (with weight tying to the input embedding), compute the language
modeling loss, and implement autoregressive text generation.

LayerNorm, CausalSelfAttention, MLP and Block below are given as working
solutions (HW1-3) so this assignment can focus on how they compose into
the full model, and on turning next-token logits into actual generated
text.

TODO: GPT.__init__ (assemble everything), GPT.forward(), GPT.generate()
"""

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
    """Given -- solved in HW3."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Given -- solved in HW2."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

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
    """Given -- solved in HW3."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Given -- solved in HW3."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO: create the following submodules:
        #   self.wte   = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        #   self.wpe   = nn.Embedding(cfg.block_size, cfg.n_embd)
        #   self.drop  = nn.Dropout(cfg.dropout)
        #   self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        #   self.ln_f  = LayerNorm(cfg.n_embd, cfg.bias)
        #   self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        #
        # Then TIE the output head to the token embedding:
        #   self.lm_head.weight = self.wte.weight
        #
        # Why: at this model scale the vocab embedding table can be
        # 20-35% of ALL parameters (see the course's configs.py). Tying
        # means the output head reuses that same matrix instead of
        # learning a second one -- roughly halving that cost, and
        # empirically this doesn't hurt (and often helps) quality.
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx: (B, T) token ids, T <= block_size
        targets: optional (B, T) token ids (idx shifted by one position)

        Steps:
          1. pos = torch.arange(T, device=idx.device)
          2. x = self.drop(self.wte(idx) + self.wpe(pos))   (same as HW1)
          3. pass x through each block in self.blocks, in order
          4. x = self.ln_f(x)
          5. logits = self.lm_head(x)   -> (B, T, vocab_size)
          6. if targets is not None: loss = F.cross_entropy(
                 logits.view(-1, logits.size(-1)), targets.view(-1))
             else: loss = None

        Returns: (logits, loss)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        # TODO: implement
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive sampling loop. Repeat max_new_tokens times:
          1. Crop idx to the last block_size tokens: idx_cond = idx[:, -self.cfg.block_size:]
             (the model can't take a longer sequence than block_size)
          2. logits, _ = self(idx_cond)
          3. Take only the LAST position's logits: logits[:, -1, :], divide by temperature
          4. If top_k is not None: keep only the top_k logits, set the rest to -inf
             (torch.topk gives you the values; anything below the k-th
             largest value should become -float("inf") before softmax)
          5. probs = F.softmax(logits, dim=-1)
          6. idx_next = torch.multinomial(probs, num_samples=1)
          7. idx = torch.cat((idx, idx_next), dim=1)

        Returns: idx, extended by max_new_tokens new tokens -> shape (B, T + max_new_tokens)
        """
        # TODO: implement
        raise NotImplementedError
