"""PG19 books (deepmind/pg19, train split) for the long-context extension mix, as Megatron .bin/.idx.

The dataset is streamed in a fixed shuffle (seed 0, buffer 10,000) and the first 12,200 books of at least 90,000
characters are skipped: they are the held-out books the paper evaluates on, so no evaluation book is trained on.

    python data/prepare_pg19.py --out-dir data/pg19 --train-tokens 210e6      # 400M extension (10 % of 2.1B)
    python data/prepare_pg19.py --out-dir data/pg19 --train-tokens 720e6      # 1.3B extension (10 % of 7.2B)

Writes <out-dir>/train.{bin,idx} (documents = BOS ... EOS, as the pretraining data)."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.megatron import MegatronWriter  # noqa: E402

RESERVED_BOOKS = 12_200
MIN_CHARS = 90_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-tokens", type=float, required=True)
    ap.add_argument("--tokenizer", default="fla-hub/transformer-1.3B-100B")  # Llama-2 32k vocabulary
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    ds = load_dataset("deepmind/pg19", split="train", streaming=True).shuffle(seed=args.seed, buffer_size=10_000)

    os.makedirs(args.out_dir, exist_ok=True)
    train = MegatronWriter(os.path.join(args.out_dir, "train"))
    reserved, batch = 0, []

    def flush():
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            train.add_document([bos] + ids + [eos])
        batch.clear()

    for ex in ds:
        if reserved < RESERVED_BOOKS:
            reserved += len(ex["text"]) >= MIN_CHARS
            continue
        if train.total_tokens >= args.train_tokens:
            break
        batch.append(ex["text"])
        if len(batch) == 8:
            flush()
            if len(train.sizes) % 800 == 0:
                print(f"train: {len(train.sizes)} books, {train.total_tokens:,} tokens", flush=True)
    flush()
    train.finalize()
    print(f"train: {len(train.sizes)} books, {train.total_tokens:,} tokens")


if __name__ == "__main__":
    main()
