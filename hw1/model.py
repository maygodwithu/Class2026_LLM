"""
Adapted from real nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/model.py, MIT).
TODO: implement GPT.forward()
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
        # TODO: look up the positional embedding for `pos`, add it to
        # tok_emb, apply self.transformer.drop, and return it.
        raise NotImplementedError
