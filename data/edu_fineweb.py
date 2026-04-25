import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download and tokenize a subset of fineweb-edu.")
    parser.add_argument(
        '--tokens', 
        type=int, 
        default=1_000_000_000, 
        help='Approximate number of tokens to extract (default: 1,000,000,000 or 1B)'
    )
    args = parser.parse_args()

    # The sample-10BT dataset has ~9.67 million rows for ~10 billion tokens.
    # This yields an average of about 1,034 tokens per document.
    TOKENS_PER_ROW = 1034
    
    # Calculate how many rows we need to hit the requested token count
    rows_to_select = int(args.tokens / TOKENS_PER_ROW)
    
    # Cap the rows so we don't accidentally ask for more data than the 10BT sample contains
    MAX_ROWS = 9670000 
    if rows_to_select > MAX_ROWS:
        print(f"Warning: Requested tokens exceed the 10BT dataset size. Capping at {MAX_ROWS:,} rows.")
        rows_to_select = MAX_ROWS

    print(f"Loading the dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", num_proc=num_proc_load_dataset)

    print(f"Subsetting to ~{args.tokens:,} tokens ({rows_to_select:,} rows)...")
    dataset = dataset.select(range(rows_to_select))

    print("Splitting into train and val...")
    # Dynamically scale the test size to always keep around 4,000 to 5,000 validation examples,
    # regardless of how many tokens you request.
    target_val_rows = 4000
    test_ratio = target_val_rows / rows_to_select
    # Ensure test_ratio makes sense (e.g., if you request a very small token count)
    test_ratio = min(max(test_ratio, 0.0005), 0.2) 

    split_dataset = dataset.train_test_split(test_size=test_ratio, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing the splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=dataset.column_names, 
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"Finished writing {filename}. Exact total tokens: {arr_len:,}")