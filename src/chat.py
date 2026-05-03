import torch
import tiktoken
from gpt import GPT 

# 1. Setup Constants
IM_START = 50257
IM_END = 50258
NEWLINE = 198

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading KhopeshAi to {device}...")

# Point this to your FINAL checkpoint (e.g., step 2000)
checkpoint_path = "out_gpt2/ckpt_21999.pt" 
model = GPT.load_custom_checkpoint(checkpoint_path, device=device)
model.eval()

enc = tiktoken.get_encoding('gpt2')

def generate_response(user_prompt, max_tokens=256, temperature=0.7, top_k=40):
    # 2. Construct the ChatML structure directly using token IDs
    system_msg = "You are a helpful and intelligent AI assistant, named GPT2-155M parameter."
    
    # System Message
    prompt_tokens = [IM_START] + enc.encode("system\n") + enc.encode(system_msg) + [IM_END, NEWLINE]
    
    # User Message
    prompt_tokens += [IM_START] + enc.encode("user\n") + enc.encode(user_prompt) + [IM_END, NEWLINE]
    
    # Assistant Header (Cue the model to start answering)
    prompt_tokens += [IM_START] + enc.encode("assistant\n")
    
    # Convert to tensor
    xgen = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 3. Generate
    with torch.no_grad():
        generated_idx = model.generate(xgen, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    # 4. Decode Safely
    generated_tokens = generated_idx[0].tolist()
    
    # Slice off the prompt so we only look at the new generated tokens
    new_tokens = generated_tokens[len(prompt_tokens):]
    
    chunk = []
    for t in new_tokens:
        if t == IM_END: # The model has decided it is done talking!
            break       
        if t == IM_START: # Shouldn't happen, but just in case
            continue
        chunk.append(t)
        
    return enc.decode(chunk).strip()

print("\n=== GPT-2 155M Local Chat ===")
print("Type 'quit' or 'exit' to end the session.")
print("-" * 55)

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nShutting down GPT-2. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        response = generate_response(user_input)
        print(f"\nAI Assistant: {response}")
        print("-" * 55)
        
    except KeyboardInterrupt:
        print("\nForce quitting...")
        break