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
