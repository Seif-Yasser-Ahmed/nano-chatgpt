from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    use_checkpoint: bool = False

