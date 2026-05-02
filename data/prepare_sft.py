import json
import numpy as np
import tiktoken
from tqdm import tqdm

# We will use unused token slots in your 50304 vocab buffer for the special tokens
IM_START = 50257
IM_END = 50258
NEWLINE = 198 # GPT2 encoding for '\n'

enc = tiktoken.get_encoding("gpt2")

def encode_chatml(conversation):
    x_tokens = []
    y_tokens = []
    
    for message in conversation:
        role = message["from"] # 'human', 'gpt', or 'system'
        value = message["value"]
        
        # Format: <|im_start|>role\ncontent<|im_end|>\n
        header = f"{'user' if role == 'human' else 'assistant' if role == 'gpt' else 'system'}\n"
        
        header_tokens = [IM_START] + enc.encode(header, allowed_special="all")
        content_tokens = enc.encode(value, allowed_special="all")
        footer_tokens = [IM_END, NEWLINE]
        
        msg_tokens = header_tokens + content_tokens + footer_tokens
        x_tokens.extend(msg_tokens)
        
        # Masking logic: If it's the assistant, we want the model to learn it. 
        # If it's the user/system, we mask it with -100 so loss is ignored.
        if role == 'gpt':
            # We train on the assistant's content and the footer (so it learns to stop)
            y_mask = [-100] * len(header_tokens) + content_tokens + footer_tokens
        else:
            y_mask = [-100] * len(msg_tokens)
            
        y_tokens.extend(y_mask)
        
    return x_tokens, y_tokens

print("Processing SFT data...")
all_x = []
all_y = []

# Load the 100K JSONL you generated earlier
with open("openhermes_155M_sft_100k.jsonl", "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        # OpenHermes uses 'conversations' key
        x, y = encode_chatml(data["conversations"])
        all_x.extend(x)
        all_y.extend(y)

# We need to shift y by 1 relative to x for next-token prediction
all_x = all_x[:-1]
all_y = all_y[1:]

# Save as memmap binaries. 
# x can be uint16 (0-65535), but y must be int16 to support -100
print(f"Total tokens: {len(all_x)}")
x_arr = np.array(all_x, dtype=np.uint16)
y_arr = np.array(all_y, dtype=np.int16)

x_arr.tofile("data/sft_x.bin")
y_arr.tofile("data/sft_y.bin")
print("Saved data/sft_x.bin and data/sft_y.bin")