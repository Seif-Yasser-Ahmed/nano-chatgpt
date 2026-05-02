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
        self.split = split
        
        if data_path is not None and data_path.endswith('.txt'):
            if process_rank == 0:
                print(f"Loading raw text file for local testing: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            
        else:
            assert split in {'train', 'val', 'sft'}, f"Split must be 'train', 'val', or 'sft', got {split}"
            
            if split == 'sft':
                # Load the pre-processed SFT binaries
                x_bin_path = 'data/sft_x.bin'
                y_bin_path = 'data/sft_y.bin'
                if process_rank == 0:
                    print(f"Loading SFT split via memmap...")
                
                # IMPORTANT: y is int32 to hold 50258 and -100 simultaneously
                self.tokens_x = np.memmap(x_bin_path, dtype=np.uint16, mode='r')
                self.tokens_y = np.memmap(y_bin_path, dtype=np.int32, mode='r') 
                self.tokens = self.tokens_x # for global length calculations
            else:
                # Pre-training fallback
                bin_path = f'data/{split}.bin'
                if process_rank == 0:
                    print(f"Loading {split} split via memmap from: {bin_path}")
                self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
                self.tokens_x = self.tokens
                self.tokens_y = self.tokens 

        if process_rank == 0:
            print(f"Dataset length in tokens: {len(self.tokens):,}")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_index = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        
        # SFT Logic: The data is pre-aligned, pre-padded, and pre-shifted
        if self.split == 'sft':
            buf_x = self.tokens_x[self.current_index : self.current_index + B * T]
            buf_y = self.tokens_y[self.current_index : self.current_index + B * T]
            
            # Convert to tensors directly
            x = torch.tensor(buf_x.astype(np.int64), dtype=torch.long).view(B, T)
            y = torch.tensor(buf_y.astype(np.int64), dtype=torch.long).view(B, T)
            
        # Pre-training Logic: We must shift y by 1 manually
        else:
            buf = self.tokens[self.current_index : self.current_index + B * T + 1]
            buf_tensor = torch.tensor(buf.astype(np.int64), dtype=torch.long)
            x = buf_tensor[:-1].view(B, T)
            y = buf_tensor[1:].view(B, T)
        
        # Advance index by the total tokens processed by all GPUs/processes
        self.current_index += B * T * self.num_processes
        
        # Reset if we hit the end of the dataset
        if self.current_index + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_index = self.B * self.T * self.process_rank
            
        return x, y