from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from attention import CausalSelfAttention, FlashAttention
from mlp import MLP, ModernMLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # self.attn = CausalSelfAttention(config)
        self.attn = FlashAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # self.mlp = MLP(config)
        if config.use_checkpoint:
            self.mlp = MLP(config)
        else:
            self.mlp = ModernMLP(config)

    def forward(self, x,kv_cache=None):  # x-->layer norm->self-attn->residual connection->layer norm->MLP->residual connection
        attn_out,kv_cache_b=self.attn(self.ln_1(x),kv_cache=kv_cache)
        x = x + attn_out
        # x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x,kv_cache_b
