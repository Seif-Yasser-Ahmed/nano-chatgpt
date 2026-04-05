from gpt import GPT
import tiktoken
import torch
from torch.nn import functional as F
num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained('gpt2')
# print("did it work?")
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, Iam a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x=tokens.to(device)

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
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">", decoded)
