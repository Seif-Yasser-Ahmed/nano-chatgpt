import torch
import tiktoken
from gpt import GPT # Adjust import path if needed

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model to {device}...")

# Point this to your FINAL checkpoint
checkpoint_path = "out_gpt2/ckpt_19999.pt" 
model = GPT.load_custom_checkpoint(checkpoint_path, device=device)
model.eval()

# If using torch.compile in training, we don't necessarily need it for quick inference, 
# but you can add it here if you want max speed.
enc = tiktoken.get_encoding('gpt2')

def generate_response(user_prompt, max_tokens=256, temperature=0.3, top_k=10):
    # 2. Format as ChatML
# Add the system prompt to the very beginning
    system_msg = "You are KhopeshAi, a helpful and intelligent AI assistant."
    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"    
    # 3. Encode safely with our custom special tokens
    prompt_replaced = full_prompt.replace("<|im_start|>", " IM_START ").replace("<|im_end|>", " IM_END ")
    tokens = enc.encode(prompt_replaced, allowed_special="all")
    tokens = [50257 if t == enc.encode(" IM_START ")[0] else 50258 if t == enc.encode(" IM_END ")[0] else t for t in tokens]
    
    xgen = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 4. Generate
    with torch.no_grad():
        # Using the generate function from your gpt.py
        generated_idx = model.generate(xgen, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    # 5. Decode safely and stop at <|im_end|>
    generated_tokens = generated_idx[0].tolist()
    
    # Slice off the prompt so we only print the new generated tokens
    new_tokens = generated_tokens[len(tokens):]
    
    chunk = []
    for t in new_tokens:
        if t == 50258: # <|im_end|> token
            break      # The model has decided it is done talking! Stop processing.
        if t == 50257: # <|im_start|> (Shouldn't happen, but just in case)
            continue
        chunk.append(t)
        
    return enc.decode(chunk)

print("\n=== KhopeshAi 155M Local Chat (type 'quit' to exit) ===")
print("-" * 55)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
        
    response = generate_response(user_input)
    print(f"\nAssistant: {response}")
    print("-" * 55)