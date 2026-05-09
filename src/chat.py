import argparse
import torch
import tiktoken
from gpt import GPT

# 1. Setup Constants
IM_START = 50257
IM_END = 50258
NEWLINE = 198


def generate_response(model, enc, device, user_prompt, system_msg, max_tokens, temperature, top_k):
    prompt_tokens = [
        IM_START] + enc.encode("system\n") + enc.encode(system_msg) + [IM_END, NEWLINE]

    prompt_tokens += [IM_START] + \
        enc.encode("user\n") + enc.encode(user_prompt) + [IM_END, NEWLINE]

    prompt_tokens += [IM_START] + enc.encode("assistant\n")

    xgen = torch.tensor(prompt_tokens, dtype=torch.long,
                        device=device).unsqueeze(0)

    # 3. Generate
    with torch.no_grad():
        generated_idx = model.generate(
            xgen,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    generated_tokens = generated_idx[0].tolist()

    new_tokens = generated_tokens[len(prompt_tokens):]

    chunk = []
    for t in new_tokens:
        if t == IM_END:  # The model has decided it is done talking!
            break
        if t == IM_START:  # Shouldn't happen, but just in case
            continue
        chunk.append(t)

    return enc.decode(chunk).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run a local chat session with a fine-tuned GPT model.")

    parser.add_argument("--model_path", type=str, default="out_gpt2/10B/finetuned/ckpt_21999.pt",
                        help="Path to the model checkpoint.")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to load the model on (e.g., 'cuda', 'cpu', 'mps').")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate per response.")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature for generation (higher = more random).")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful and intelligent AI assistant, named GPT2-155M parameter.",
                        help="The system prompt to condition the AI's behavior.")

    args = parser.parse_args()

    print(f"Loading GPT-2 from '{args.model_path}' to {args.device}...")

    model = GPT.load_custom_checkpoint(args.model_path, device=args.device,visualize_attention=False)
    model.eval()
    enc = tiktoken.get_encoding('gpt2')

    print("\n=== GPT-2 Local Chat ===")
    print(f"System Prompt: '{args.system_prompt}'")
    print(
        f"Params: Temp={args.temp}, TopK={args.top_k}, MaxTokens={args.max_tokens}")
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

            response = generate_response(
                model=model,
                enc=enc,
                device=args.device,
                user_prompt=user_input,
                system_msg=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k
            )

            print(f"\nAI Assistant: {response}")
            print("-" * 55)

        except KeyboardInterrupt:
            print("\nForce quitting...")
            break


if __name__ == "__main__":
    main()
