import os
import pathlib
import submitit
from tqdm import tqdm
import glob

import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

input_paths = glob.glob('/data/c-cye/assignment4-data/cc_filtered/*.txt')
# sort in order of creation time
input_paths.sort(key=lambda x: os.path.getctime(x))

output_dir = "/data/c-cye/assignment4-data/cc_tokenized"
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Tokenizer loaded")


def tokenize_file(input_path: str):
    """Process a single file and save the tokenized output"""
    # create output path
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename.replace('.txt', '.bin'))

    if os.path.exists(output_path):
        print(f"Skipping {input_path} because it already exists")
        return 0

    print(f"Tokenizing {input_path}...")
    
    with open(input_path) as f:
        lines = f.readlines()
    
    # tokenize all lines in this file
    results = []
    for line in lines:
        results.append(tokenizer.encode(line))
    
    # flatten all token IDs into a single array
    all_ids = [token_id for sublist in results for token_id in sublist]
    # truncate up to last <|endoftext|>
    last_eos = len(all_ids) - 1 - all_ids[::-1].index(tokenizer.eos_token_id)
    all_ids = all_ids[:last_eos + 1] # include the <|endoftext|>

    print(f"Tokenized {input_path} into {len(all_ids)} tokens")
    
    # save as binary file
    ids_array = np.array(all_ids, dtype=np.uint16)
    # overwrite existing file

    if os.path.exists(output_path):
        os.remove(output_path)
    ids_array.tofile(output_path)
    
    return len(all_ids)

def process_files_parallel():
    """process all files using multiprocessing"""
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # process files in parallel
        results = list(tqdm(
            pool.imap(tokenize_file, input_paths),
            total=len(input_paths),
            desc="Processing files"
        ))
    
    total_tokens = sum(results)
    print(f"Total tokens processed: {total_tokens}")

if __name__ == "__main__":
    process_files_parallel()