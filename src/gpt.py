from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block
from gpt_config import GPTConfig
import inspect

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
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std=0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std = std * (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx,targets=None):
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
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits,loss #expected loss at init is -ln(1/vocab_size)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config_args['use_checkpoint'] = True
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict={pn: p for pn, p in self.named_parameters()}
        param_dict={pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params=[p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n,p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        print(f"num decay paraneter tensor: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decay paraneter tensor: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        fused_available="fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and "cuda" in device
        print(f"Fused AdamW available: {fused_available}, using fused AdamW: {use_fused}")
        optimizer=torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer