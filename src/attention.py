import torch
import torch.nn as nn
from torch.nn import functional as F
import math
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

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C) each
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply KV Cache logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
            
        current_kv_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) * \
            (1.0/math.sqrt(k.size(-1)))  # (B, nh, T, T_k)
            
        # Only apply the causal mask during the prefill phase (when kv_cache is None)
        if kv_cache is None:
            att = att.masked_fill(
                self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, nh, T, T)
                
        att = F.softmax(att, dim=-1)  # (B, nh, T, T_k)
        
        # Save attention weights for BertViz!
        self.saved_attention = att 
        
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)  # (B, T, C)
        if not self.config.use_checkpoint:
            y = self.resid_dropout(y)
            
        return y, current_kv_cache

class FlashAttention(nn.Module):
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

    def forward(self, x,kv_cache=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C) each
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
        
        current_kv_cache = (k, v)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=(kv_cache is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)  # (B, T, C)
        if not self.config.use_checkpoint:
            y = self.resid_dropout(y)
        return y,current_kv_cache
    # pass
