import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Constants for your architecture
MAX_T = 1024
IM_START = 50257
IM_END = 50258
NEWLINE = 198
PAD_TOKEN = -100

enc = tiktoken.get_encoding("gpt2")

def format_alpaca_to_chatml(example):
    """Converts Alpaca instruction/input/output into ChatML format."""
    instruction = example['instruction']
    inp = example.get('input', '')
    output = example['output']
    
    # Combine instruction and input if input exists
    user_text = f"{instruction}\n{inp}" if inp else instruction
    
    return [
        {"from": "system", "value": "You are a helpful, logical, and concise AI assistant."},
        {"from": "human", "value": user_text},
        {"from": "gpt", "value": output}
    ]

def encode_chatml(conversation):
    x_tokens = []
    y_tokens = []
    
    for message in conversation:
        role = message["from"] 
        value = message["value"]
        
        header = f"{'user' if role == 'human' else 'assistant' if role == 'gpt' else 'system'}\n"
        header_tokens = [IM_START] + enc.encode(header, allowed_special="all")
        content_tokens = enc.encode(value, allowed_special="all")
        footer_tokens = [IM_END, NEWLINE]
        
        msg_tokens = header_tokens + content_tokens + footer_tokens
        x_tokens.extend(msg_tokens)
        
        if role == 'gpt':
            # Train on assistant content and the stop tags
            y_mask = [-100] * len(header_tokens) + content_tokens + footer_tokens
        else:
            # Mask user/system prompts
            y_mask = [-100] * len(msg_tokens)
        y_tokens.extend(y_mask)
            
    return x_tokens, y_tokens

if __name__ == "__main__":
    print("Downloading yahma/alpaca-cleaned...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    final_x = []
    final_y = []
    
    print(f"Processing {len(dataset)} examples...")
    for example in tqdm(dataset):
        chatml_conv = format_alpaca_to_chatml(example)
        x, y = encode_chatml(chatml_conv)
        
        # FILTER: Only keep if it fits in your context window
        if len(x) <= MAX_T:
            padding_len = MAX_T - len(x)
            
            x_padded = x + [NEWLINE] * padding_len
            y_padded = y + [-100] * padding_len
            
            final_x.append(x_padded)
            final_y.append(y_padded)

    print(f"Kept {len(final_x)} complete conversations out of {len(dataset)}.")
    
    # Leave X as uint16 (it only has positive token IDs)
    x_arr = np.array(final_x, dtype=np.uint16)
    
    # CHANGE Y to int32 so it can hold both 50258 and -100
    y_arr = np.array(final_y, dtype=np.int32)

    # Shift Y: Predict the next token (also changed to int32 here)
    y_shifted = np.concatenate([y_arr[:, 1:], np.full((y_arr.shape[0], 1), -100, dtype=np.int32)], axis=1)

    # Shift Y: Predict the next token
    # y_shifted = np.concatenate([y_arr[:, 1:], np.full((y_arr.shape[0], 1), -100, dtype=np.int16)], axis=1)

    os.makedirs("data", exist_ok=True)
    x_arr.tofile("data/sft_x.bin")
    y_shifted.tofile("data/sft_y.bin")
    print("Saved clean, padded binaries to data/sft_x.bin and data/sft_y.bin")