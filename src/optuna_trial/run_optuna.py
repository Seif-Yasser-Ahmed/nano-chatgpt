import os
import sys
import math
import torch
import optuna
import logging
from torch.nn import functional as F

# Add the parent 'src' directory to the Python path so we can import your custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt import GPT
from gpt_config import GPTConfig
from data_loader import DataLoaderLite

# Constants for the Sweep
TRIAL_STEPS = 250      # Roughly 5 minutes of compute on a strong GPU
EVAL_ITERS = 20        # How many batches to use for validation loss
B = 16                 # Micro-batch size
T = 1024               # Context length
GRAD_ACCUM_STEPS = 8   # Keeps total batch size smaller for fast iteration (~131k tokens)

def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Suggest Hyperparameters
    max_lr = trial.suggest_float("max_lr", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.2)
    warmup_steps = trial.suggest_int("warmup_steps", 20, 100) 
    min_lr_factor = trial.suggest_float("min_lr_factor", 0.01, 0.1)
    
    min_lr = max_lr * min_lr_factor

    # 2. Initialize Model & Data[cite: 10, 11, 16]
    config = GPTConfig(vocab_size=50304) 
    model = GPT(config)
    model.to(device)
    
    if 'cuda' in device:
        model = torch.compile(model)
        
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device)
    
    # Using 'train' and 'val' splits for pretraining[cite: 16]
    train_loader = DataLoaderLite(B=B, T=T, split='train')
    val_loader = DataLoaderLite(B=B, T=T, split='val')

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > TRIAL_STEPS:
            return min_lr
        decay_ratio = (it - warmup_steps) / (TRIAL_STEPS - warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff

    # 3. Training Loop
    for step in range(TRIAL_STEPS):
        model.train()
        optimizer.zero_grad()
        
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device != 'cpu' else torch.float32):
                logits, loss = model(x, y)
                
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        
        # Report intermediate loss to Optuna so it can prune (kill) terrible runs early
        if step % 50 == 0:
            trial.report(loss.item(), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # 4. Final Evaluation
    model.eval()
    val_loss_accum = 0.0
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            x_val, y_val = val_loader.next_batch()
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device != 'cpu' else torch.float32):
                logits, loss = model(x_val, y_val)
            val_loss_accum += loss.item()
            
    final_val_loss = val_loss_accum / EVAL_ITERS
    return final_val_loss

if __name__ == "__main__":
    # Ensure logs show up nicely
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    # --- THE UPGRADE: SQLite Database ---
    study_name = "gpt2_155M_pretrain_sweep"
    storage_name = "sqlite:///optuna_journal.db"
    
    # Create or load the study. We want to MINIMIZE the validation loss.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # <--- This lets you pause and resume!
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50) 
    )
    
    print("\n--- Starting Optuna Hyperparameter Sweep ---")
    print(f"Target: {TRIAL_STEPS} steps per trial.")
    print(f"Saving all history to: {storage_name}")
    
    # Run 20 different trials to find the best combination
    study.optimize(objective, n_trials=20)
    
    print("\n--- SWEEP COMPLETE ---")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value:.4f}")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # --- EXPORT EVERYTHING ---
    # 1. Save the best trial text file
    with open("optuna_best_result.txt", "w") as f:
        f.write(f"Best Validation Loss: {trial.value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
            
    # 2. Dump every single trial to a CSV for easy viewing
    try:
        df = study.trials_dataframe()
        df.to_csv("optuna_all_trials.csv", index=False)
        print("\nSuccessfully saved all trial data to 'optuna_all_trials.csv'")
    except ImportError:
        print("\nNote: Install 'pandas' (pip install pandas) to automatically export the CSV log.")