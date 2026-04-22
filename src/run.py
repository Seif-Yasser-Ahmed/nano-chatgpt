from gpt import GPT
from gpt_config import GPTConfig
import tiktoken
import torch
from torch.nn import functional as F
from data_loader import DataLoaderLite
USE_PRETRAINED = False
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


train_loader=DataLoaderLite(B=4,T=32)
model=GPT(GPTConfig())
# model.eval()
model.to(device)

optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x,y=train_loader.next_batch()
    x=x.to(device)
    y=y.to(device)
    optimizer.zero_grad()
    logits,loss=model(x,y)
    # import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item():.4f}")
# print(logits.shape)



if USE_PRETRAINED:
    enc=tiktoken.get_encoding('gpt2')
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
