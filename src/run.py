import time
from gpt import GPT
from gpt_config import GPTConfig
import tiktoken
import torch
from torch.nn import functional as F
from data_loader import DataLoaderLite
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.distributed as dist
USE_PRETRAINED = False

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f"Using device: {device}")
# tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# print(f"tokens per iteration will be: {tokens_per_iter:,}")



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


out_dir = 'out-gpt2'
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# 2. Fix the max_steps bug
total_batch_size = 524288
B = 2
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "total_batch_size must be divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_path='test_data/input.txt')

val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_path='test_data/input.txt') # Or a separate val.txt
torch.set_float32_matmul_precision('high')


model = GPT(GPTConfig(vocab_size=50304))
# model.eval()
model.to(device)

# if torch.cuda.device_count() > 1:
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(model)

# model = torch.compile(model) if hasattr(torch, 'compile') else model

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=71
max_steps=1907

def get_lr(it):
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    if it>max_steps:
        return min_lr
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1+math.cos(math.pi*decay_ratio))
    return min_lr+(max_lr-min_lr)*coeff

# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4,betas=(0.9,0.95),eps=1e-8)
optimizer=raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
num_steps = len(train_loader.tokens) // (train_loader.B * train_loader.T)
if master_process:
    print(f"Training for 1 epoch ({num_steps} steps)")

# 4. Define evaluation intervals
eval_interval = 250
eval_iters = 20 # Number of batches to average for validation loss

for step in range(max_steps):
    
    # --- EVALUATION & CHECKPOINTING PHASE ---
    if step % eval_interval == 0 or step == max_steps - 1:
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for _ in range(eval_iters):
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16 if device != 'cpu' else torch.float32):
                    logits, loss = model(x_val, y_val)
                val_loss_accum += loss.detach()
        
        val_loss_accum = val_loss_accum / eval_iters
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
        if master_process:
            print(f"step {step} | val loss: {val_loss_accum.item():.4f}")
            # Save the checkpoint
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'val_loss': val_loss_accum.item(),
            }
            checkpoint_name = f'ckpt_{step:05d}.pt'
            torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
            print(f"saved checkpoint to {out_dir}/{checkpoint_name}")
            
        model.train()
    # -----------------------------------------

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device != 'cpu' else torch.float32):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    
    t1 = time.time()
    dt = (t1 - t0)
    
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / dt
    
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.4f} | lr={lr:.6f} | norm {norm:.4f} | time: {dt*1000:.2f} ms | tok/s: {tokens_per_second:.2f}")

if ddp:
    destroy_process_group()

if USE_PRETRAINED:
    enc = tiktoken.get_encoding('gpt2')
    num_return_sequences = 5
    max_length = 30
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)

    tokens = enc.encode("Hello, Iam a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
