from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        if config.use_checkpoint:
            self.mlp = MLP(config)
        else:
            self.mlp = ModernMLP(config)

    def forward(self, x):  # x-->layer norm->self-attn->residual connection->layer norm->MLP->residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = config.n_head
        self.config = config
        # linear layer to compute query, key, value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # linear layer to project the output of the attention mechanism back to the original embedding dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        if not config.use_checkpoint:
            # dropout layer to prevent overfitting in the attention mechanism
            self.attn_dropout = nn.Dropout(config.dropout)
            # dropout layer to prevent overfitting in the residual connection
            self.resid_dropout = nn.Dropout(config.dropout)
        else:
            self.resid_dropout = nn.Identity()
        self.n_head = config.n_head
        self.n_embed = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C) each
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * \
            (1.0/math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, nh, T, T)
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)  # (B, T, C)
        if not self.config.use_checkpoint:
            y = self.resid_dropout(y)
        return y
    # pass


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
