from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    use_checkpoint: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            # token embeddings in the original Attention is all you need paper
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # position embeddings in the original Attention is all you need paper
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            # dropout layer that randomly sets a fraction of the input units to 0 during training to prevent overfitting
            'drop': nn.Dropout(config.dropout) if not config.use_checkpoint else nn.Identity(),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        # final classification head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T},model block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long,
                           device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.transformer.wpe(pos)  # (1, T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
