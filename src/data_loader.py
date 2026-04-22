import tiktoken
import torch
class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T
        with open('test_data/input.txt', 'r') as f:
            text = f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"Dataset length in tokens: {len(self.tokens)}")
        print("1 epoch = %d batches" % (len(self.tokens) // (B*T)))
        print("1 epoch = %d tokens" % (len(self.tokens) // (B*T)*B*T))
        print("1 epoch =%d iterations" % (len(self.tokens) // (B*T)))
        self.current_index=0
    
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_index:self.current_index+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_index+=B*T
        if self.current_index+B*T+1>=len(self.tokens):
            self.current_index=0
        return x,y
