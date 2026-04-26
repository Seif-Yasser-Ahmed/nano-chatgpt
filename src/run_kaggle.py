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
import argparse

def run(manual_seed=1337, out_dir='out_gpt2', total_batch_size=524288, B=64, T=1024, max_lr=6e-4, min_lr=None, min_lr_factor=0.1, warmup_steps=715, max_steps=19073, eval_interval=100, eval_iters=20, resume_ckpt=None):
    enc = tiktoken.get_encoding('gpt2')
    
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
    
        print(f"Using device: {device}")

    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    assert total_batch_size % (B*T*ddp_world_size) == 0, "total_batch_size must be divisible by B*T*ddp_world_size"
    grad_accum_steps = total_batch_size // (B*T*ddp_world_size)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    min_lr = max_lr * min_lr_factor if min_lr is None else min_lr

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    start_step = 0
    if resume_ckpt and os.path.isfile(resume_ckpt):
        if master_process:
            print(f"Loading checkpoint from {resume_ckpt}...")
            
        checkpoint = torch.load(resume_ckpt, map_location=device)
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_step = checkpoint['step'] + 1
        
        batches_to_skip = start_step * grad_accum_steps
        if master_process:
            print(f"Fast-forwarding data loader by {batches_to_skip} micro-batches. This may take a few seconds...")
            
        for _ in range(batches_to_skip):
            train_loader.next_batch()
            
        if master_process:
            print(f"Resumed successfully! Continuing from step {start_step}.")

    num_steps = len(train_loader.tokens) // (train_loader.B * train_loader.T)
    if master_process and start_step == 0:
        print(f"Training for 1 epoch ({num_steps} steps)")

    # Loop modified to start at `start_step` instead of 0
    for step in range(start_step, max_steps):
        
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

        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, Iam a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_props, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_props, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} | sample {i} | {decoded}")
                if master_process:
                    with open(os.path.join(out_dir, "sampled", f'step{step}_sample{i}.txt'), 'w') as f:
                        f.write(decoded)
        
        model.train() 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model")

    parser.add_argument("--manual_seed", type=int, default=1337)
    parser.add_argument("--out_dir", type=str, default="out_gpt2")
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--min_lr_factor", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--max_steps", type=int, default=19073)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint .pt file to resume from")

    args = parser.parse_args()

    run(
        manual_seed=args.manual_seed,
        out_dir=args.out_dir,
        total_batch_size=args.total_batch_size,
        B=args.B,
        T=args.T,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        min_lr_factor=args.min_lr_factor,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        resume_ckpt=args.resume_ckpt
    )