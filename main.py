import argparse
import torch
import os
from src.gpt import GPT

def main():
    parser = argparse.ArgumentParser(description="Generate text completions using your custom GPT model")
    
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to the custom .pt checkpoint")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--prompt", type=str, help="A single prompt string to complete")
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing prompts (one per line)")
    
    parser.add_argument("-m", "--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature for sampling (lower = more deterministic)")
    parser.add_argument("-k", "--top_k", type=int, default=50, help="Top-K sampling (None to disable)")
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPT.load_custom_checkpoint(args.ckpt, device=device)

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(prompts)} prompts from {args.file}")
    else:
        prompts = args.prompt

    print(f"\nGenerating completions (max_new_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k})...\n")
    print("-" * 50)
    
    results = model.create_chat_completions(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    if isinstance(results, str):
        print(f"{results}")
    else:
        for i, res in enumerate(results):
            print(f"--- Prompt {i+1} ---")
            print(res)
            print("-" * 50)

if __name__ == "__main__":
    main()