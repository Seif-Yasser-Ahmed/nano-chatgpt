import os
import torch
import numpy as np
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train', data_path=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        if data_path is not None and data_path.endswith('.txt'):
            if process_rank == 0:
                print(f"Loading raw text file for local testing: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            
        else:
            assert split in {'train', 'val'}, f"Split must be 'train' or 'val', got {split}"
            bin_path = f'data/{split}.bin'
            
            if process_rank == 0:
                print(f"Loading {split} split via memmap from: {bin_path}")
                
            self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')

        if process_rank == 0:
            print(f"Dataset length in tokens: {len(self.tokens):,}")
            print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_index = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_index : self.current_index + B * T + 1]
        
        if isinstance(buf, np.ndarray):
            buf = torch.tensor(buf.astype(np.int64), dtype=torch.long)
        # If it came from a .txt file, it is already a torch tensor, so we do nothing
            
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_index += B * T * self.num_processes
        
        # Reset if the next batch would go out of bounds
        if self.current_index + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_index = self.B * self.T * self.process_rank
            
        return x, y