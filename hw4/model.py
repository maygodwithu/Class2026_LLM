"""
HW3의 MultiHeadAttention에 LayerNorm, FeedForward, shortcut을 붙여 TransformerBlock을 만들고,
그 블록을 쌓아 GPT(decoder-only)를 완성하는 파일. 교재 4장의 형태를 따릅니다.
LayerNorm과 GELU는 PyTorch에 있는 것(nn.LayerNorm, nn.GELU)을 그대로 씁니다.
HW1의 GPT 앞부분(임베딩), HW3의 MultiHeadAttention은 이미 완성되어 있습니다.
TODO 1: FeedForward.__init__()  (한 줄)
TODO 2: TransformerBlock.forward()  (두 줄)
TODO 3: GPT.__init__()  (한 줄, 층 수 정하기)
TODO 4: GPT.forward()  (두 줄)
TODO 5: GPT.generate()  (한 줄)
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
        # TODO 1: 다시 n_embd로 줄이는 Linear 층 down 만들기  (4*n_embd -> n_embd)
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
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
        # TODO 2: drop_shortcut(dropout)을 적용하고, 원래 입력(shortcut)을 더하기  (아래 피드 포워드 블록과 같은 모양)
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요

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
        # TODO 3: 블록을 몇 층 쌓을지 num_layers를 정하세요  (config에 있는 값, 교재 4.6)
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
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
        # TODO 4: final_norm을 거친 뒤 out_head를 통과시켜 logits를 만드세요 (교재 4.6)
        raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):  # 교재 4.7 (그리디 디코딩)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # 문맥이 너무 길면 뒤쪽만 사용
            logits = self(idx_cond)  # (b, t, vocab_size)
            logits = logits[:, -1, :]  # 마지막 위치의 logits만 사용: (b, vocab_size)
            # TODO 5: logits에서 가장 큰 값의 토큰을 골라 idx_next (b, 1)를 만드세요 (그리디)
            #   (torch.argmax, dim=-1, keepdim=True)
            raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
