"""
HW2의 CausalAttention(head 1개)을 multi-head로 키운 파일. 교재 3.6절의 형태를 따릅니다.
HW1의 GPT는 이미 완성되어 있고, 이번에는 GPT가 MultiHeadAttention까지 사용합니다.
TODO 1, 2: MultiHeadAttention.forward()
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
    dropout: float = 0.0


# class CausalAttention(nn.Module):  # HW2: head 1개
#     ...
#
# CausalAttention(head 1개)  ->  MultiHeadAttention(head 여러 개)으로 진화
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어 떨어져야 합니다"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # head 하나가 맡는 차원
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # head들의 출력을 합친 뒤 섞어주는 층
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # TODO 1: head 나누기  (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        #   keys, queries, values 각각  .view(b, num_tokens, self.num_heads, self.head_dim)  다음  .transpose(1, 2)
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요

        attn_scores = queries @ keys.transpose(2, 3)  # head별로 점곱
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_heads, num_tokens, head_dim)

        # TODO 2: head 합치기  (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, d_out), 그 다음 out_proj
        #   .transpose(1, 2)  ->  .contiguous().view(b, num_tokens, self.d_out)  ->  self.out_proj(...)

        self.queries, self.keys, self.values = queries, keys, values  # test에서 확인하려고 저장
        self.attn_scores, self.attn_weights = attn_scores, attn_weights
        return context_vec


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
        ))
        self.att = MultiHeadAttention(  # HW3에서 추가
            d_in=config.n_embd,
            d_out=config.n_embd,
            context_length=config.block_size,
            dropout=config.dropout,
            num_heads=config.n_head)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = self.att(x)  # HW3에서 추가
        return x
