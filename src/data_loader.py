import tiktoken
import torch
class DataLoaderLite:
    def __init__(self,B,T,process_rank=0, num_processes=1):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes
        with open('test_data/input.txt', 'r') as f:
            text = f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)

        print(f"Dataset length in tokens: {len(self.tokens)}")
        print("1 epoch = %d batches" % (len(self.tokens) // (B*T)))
        print("1 epoch = %d tokens" % (len(self.tokens) // (B*T)*B*T))
        print("1 epoch =%d iterations" % (len(self.tokens) // (B*T)))

        self.current_index=self.B*self.T*process_rank
    
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_index:self.current_index+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_index+=B*T*self.num_processes
        if self.current_index+B*T*self.num_processes+1>=len(self.tokens):
            self.current_index=self.B*self.T*self.process_rank
        return x,y
