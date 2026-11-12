"""
HW4에서 완성한 model.py (GPT decoder-only: TransformerBlock을 쌓은 GPT와 그리디 generate).
이번 과제(HW5)에서는 이 파일을 고치지 않고 가져다 씁니다.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.0


# HW3에서 완성한 MultiHeadAttention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어 떨어져야 합니다"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class FeedForward(nn.Module):  # 교재 4.3

    def __init__(self, config):
        super().__init__()
        up = nn.Linear(config.n_embd, 4 * config.n_embd)  # 4배로 늘렸다가
        act = nn.GELU(approximate="tanh")
        down = nn.Linear(4 * config.n_embd, config.n_embd)
        self.layers = nn.Sequential(up, act, down)

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):  # 교재 4.4, 4.5

    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.n_embd,
            d_out=config.n_embd,
            context_length=config.block_size,
            dropout=config.dropout,
            num_heads=config.n_head)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.drop_shortcut = nn.Dropout(config.dropout)

    def forward(self, x):
        # 어텐션 블록 + shortcut
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        self.after_attention = x  # test에서 확인하려고 저장

        # 피드 포워드 블록 + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
        ))
        # HW3의 self.att 자리에 TransformerBlock을 n_layer개 쌓습니다.
        num_layers = config.n_layer
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):  # 교재 4.7 (그리디 디코딩)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # 문맥이 너무 길면 뒤쪽만 사용
            logits = self(idx_cond)  # (b, t, vocab_size)
            logits = logits[:, -1, :]  # 마지막 위치의 logits만 사용: (b, vocab_size)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
