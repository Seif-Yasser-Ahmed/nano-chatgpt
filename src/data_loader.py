import os
import torch
import numpy as np

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # Ensure we are only asking for train or val
        assert split in {'train', 'val'}

        # Map the split directly to the files you generated earlier
        data_path = f'{split}.bin'
        
        if process_rank == 0:
            print(f"Loading {split} split via memmap from: {data_path}")
            
        # We use memmap here instead of loading everything into RAM
        self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')

        if process_rank == 0:
            print(f"Dataset length in tokens: {len(self.tokens):,}")
            print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_index = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        
        # Grab the buffer slice (instantaneous with memmap)
        buf = self.tokens[self.current_index : self.current_index + B * T + 1]
        
        # Convert uint16 numpy array to int64, then wrap in a torch.long tensor
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long)
            
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_index += B * T * self.num_processes
        
        # Reset if the next batch would go out of bounds
        if self.current_index + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_index = self.B * self.T * self.process_rank
            
        return x, y