"""
HW1의 GPT(입력 표현, 정답 포함)에 CausalAttention(head 1개)을 이어붙인 파일.
CausalAttention은 교재(밑바닥부터 만들면서 배우는 LLM) 3장의 형태를 따릅니다.
TODO: CausalAttention.forward()의 TODO 1, TODO 2
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    dropout: float = 0.0


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # TODO 1: self-attention
        #   keys, queries, values 만들기   (self.W_key, self.W_query, self.W_value)
        #   attn_scores = queries @ keys^T   (keys.transpose(1, 2))
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요

        # TODO 2: causal -- mask가 1인 위치(미래)의 attn_scores를 -inf로 채우기
        #   (attn_scores.masked_fill_, self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
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

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        return self.transformer.drop(tok_emb + pos_emb)
