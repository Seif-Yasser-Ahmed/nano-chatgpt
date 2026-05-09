import gradio as gr
import torch
import tiktoken
from gpt import GPT
from bertviz import head_view
import time
import html
# --- 1. SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IM_START = 50257
IM_END = 50258
NEWLINE = 198
enc = tiktoken.get_encoding('gpt2')

# Load your final checkpoint
print("Loading GPT2Ai...")
model = GPT.load_custom_checkpoint(
    "out_gpt2/10B/finetuned/ckpt_21999.pt", device=device)
model.eval()

# --- THE FIX: Helper function to strip Gradio 6's content blocks back to raw strings ---


def extract_text(content):
    if isinstance(content, list):
        # Join all text blocks if there are multiple, ignore file blocks
        return "".join([block.get("text", "") for block in content if block.get("type") == "text"])
    return str(content)

# --- 2. GENERATION & ATTENTION LOGIC ---


def chat_and_visualize(user_input, prior_history):
    system_msg = "You are GPT2Ai, a helpful and intelligent AI assistant."

    prompt_tokens = [
        IM_START] + enc.encode("system\n") + enc.encode(system_msg) + [IM_END, NEWLINE]

    for msg in prior_history:
        role = msg["role"]
        # Run the fix on the history data
        content_str = extract_text(msg["content"])
        prompt_tokens += [IM_START] + \
            enc.encode(f"{role}\n") + enc.encode(content_str) + \
            [IM_END, NEWLINE]

    prompt_tokens += [IM_START] + \
        enc.encode("user\n") + enc.encode(user_input) + [IM_END, NEWLINE]
    prompt_tokens += [IM_START] + enc.encode("assistant\n")

    xgen = torch.tensor(prompt_tokens, dtype=torch.long,
                        device=device).unsqueeze(0)

    generated_tokens = []
    response_text = ""

    with torch.no_grad():
        generated_idx = model.generate(
            xgen, max_new_tokens=150, temperature=0.7, top_k=40)

    new_tokens = generated_idx[0].tolist()[len(prompt_tokens):]

    for t in new_tokens:
        if t == IM_END:
            break
        if t == IM_START:
            continue
        generated_tokens.append(t)
        response_text = enc.decode(generated_tokens)

        time.sleep(0.02)
        yield response_text, "<i>Generating attention graph...</i>"

    # ... (Keep everything above this exactly the same) ...

    final_sequence = prompt_tokens + generated_tokens
    x_final = torch.tensor(
        final_sequence, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        _ = model(x_final)

    # FIX 1: Explicitly move the attention tensors to the CPU
    attentions = tuple(block.attn.saved_attention.cpu()
                       for block in model.transformer.h)

    token_strings = []
    for t in final_sequence:
        if t == IM_START:
            token_strings.append("<|im_start|>")
        elif t == IM_END:
            token_strings.append("<|im_end|>")
        else:
            token_strings.append(enc.decode([t]))

    html_output = head_view(attentions, token_strings, html_action='return')

    # THE FIX: Inject custom CSS to force the text and labels to be white
    custom_css = """
    <style>
        text { fill: white !important; font-family: sans-serif; }
        body, div, span, select { color: white !important; }
    </style>
    """
    modified_html = custom_css + html_output.data

    # Escape the modified HTML and wrap it in the iframe
    escaped_html = html.escape(modified_html)
    iframe_html = f'<iframe srcdoc="{escaped_html}" width="100%" height="800px" style="border:none;"></iframe>'

    yield response_text, iframe_html


# --- 3. GRADIO UI LAYOUT ---
with gr.Blocks() as demo:
    gr.Markdown("# ⚔️ GPT2Ai - 155M Custom Architecture")

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Message GPT2Ai...",
                             placeholder="Type here and press Enter")
            clear = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Live Attention Visualization")
            gr.Markdown(
                "Hover over tokens to see the causal attention weights.")
            attention_display = gr.HTML(
                value="<i>Waiting for inference...</i>")

    def user(user_message, history):
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot(history):
        # Run the fix on the current user message before feeding it to tiktoken
        user_message = extract_text(history[-1]["content"])

        history.append({"role": "assistant", "content": ""})

        for text_chunk, html_chunk in chat_and_visualize(user_message, history[:-2]):
            history[-1]["content"] = text_chunk
            yield history, html_chunk

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, attention_display]
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860,
                share=True, theme=gr.themes.Soft())
