"""
Tokenize validation shard (shard 6542) to flat uint16 binary for ANE validation.

Companion to tokenize_to_bin.py which excludes this shard from training data.
Output: native/data/val.bin

Usage (from autoresearch-macos repo which has the right Python deps):
    cd ~/Dev/autoresearch-macos
    uv run python ~/Dev/autoresearch-ANE/tokenize_val.py
"""

import os
import sys
import struct
import argparse
import pickle
import time

import pyarrow.parquet as pq
import tiktoken

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_SHARD = 6542
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

def load_tokenizer():
    path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(path):
        print(f"Tokenizer not found at {path}")
        print("Run: cd ~/Dev/autoresearch-macos && uv run python prepare.py")
        sys.exit(1)
    with open(path, "rb") as f:
        enc = pickle.load(f)
    print(f"Tokenizer loaded: vocab_size={enc.n_vocab}")
    return enc

def tokenize_val(output_path):
    enc = load_tokenizer()
    shard_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if not os.path.exists(shard_path):
        print(f"Validation shard not found: {shard_path}")
        print("Run: cd ~/Dev/autoresearch-macos && uv run python prepare.py")
        sys.exit(1)
    print(f"Tokenizing validation shard: {VAL_FILENAME}")

    # BOS token
    special_tokens = [f"<|reserved_{i}|>" for i in range(4)]
    bos_id = enc.encode_single_token(special_tokens[0])
    print(f"BOS token ID: {bos_id}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    total_tokens = 0
    total_docs = 0

    with open(output_path, "wb") as out:
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            token_lists = enc.encode_ordinary_batch(texts, num_threads=8)

            for tokens in token_lists:
                doc_ids = [bos_id] + tokens
                buf = struct.pack(f"<{len(doc_ids)}H", *doc_ids)
                out.write(buf)
                total_tokens += len(doc_ids)
                total_docs += 1

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path)
    print(f"\nDone: {total_tokens:,} tokens from {total_docs:,} documents")
    print(f"Output: {output_path} ({file_size / 1e6:.1f} MB)")
    print(f"Time: {elapsed:.1f}s ({total_tokens / elapsed:,.0f} tokens/sec)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "native", "data", "val.bin")

    parser = argparse.ArgumentParser(description="Tokenize validation shard to uint16 binary for ANE validation")
    parser.add_argument("--output", type=str, default=default_output, help=f"Output binary path (default: {default_output})")
    args = parser.parse_args()

    tokenize_val(args.output)
