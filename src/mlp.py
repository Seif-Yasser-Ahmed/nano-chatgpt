import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        self.gelu = nn.GELU(approximate='tanh')

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        if not self.config.use_checkpoint:
            x = self.dropout(x)
        return x


class ModernMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        gate = self.silu(self.w1(x))
        up_proj = self.w2(x)

        x = gate * up_proj
        x = self.w3(x)
        return x
