from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.block import Block
from src.gpt_config import GPTConfig
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

    def forward(self, idx,targets=None,kv_cache=None):
        B, T = idx.size()
        past_length=kv_cache[0][0].size(-2) if kv_cache is not None else 0
        assert past_length+T <= self.config.block_size, f"Cannot forward sequence of length {T},model block size is only {self.config.block_size}"
        pos = torch.arange(0, past_length+T, dtype=torch.long,
                           device=idx.device).unsqueeze(0)  # (1, T)
        
        pos_emb = self.transformer.wpe(pos)  # (1, T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        new_kv_cache=[]
        for i,block in enumerate(self.transformer.h):
            block_kv_cache = kv_cache[i] if kv_cache is not None else None
            
            x, block_kv = block(x, kv_cache=block_kv_cache)
            new_kv_cache.append(block_kv)

        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits,loss,tuple(new_kv_cache) #expected loss at init is -ln(1/vocab_size)

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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.eval() # Ensure the model is in evaluation mode
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :]
            
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                probs = F.softmax(logits, dim=-1)
                
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def load_custom_checkpoint(cls, checkpoint_path, device='cpu'):
        """
        Loads a custom trained model from a saved .pt checkpoint.
        """
        print(f"Loading checkpoint from {checkpoint_path} to {device}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Initialize with the exact same config used in run.py
        config = GPTConfig(vocab_size=50304)
        model = cls(config)
        
        state_dict = checkpoint['model']
        
        # If the model was trained with torch.compile(), PyTorch adds an '_orig_mod.' prefix.
        # We strip it out here so the dictionary keys match the base model exactly.
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Hardcode eval mode since this is for generation
        
        return model
    
    
    def create_chat_completions(self, prompts, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Takes a string or list of strings, encodes them, generates completions, 
        and returns the decoded full strings.
        """
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        device = next(self.parameters()).device
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            
        results = []
        for prompt in prompts:
            # 1. Encode prompt
            tokens = enc.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
            
            # 2. Generate
            generated_idx = self.generate(
                idx, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
            
            # 3. Decode back to string
            generated_text = enc.decode(generated_idx[0].tolist())
            results.append(generated_text)
            
        return results[0] if is_single else results