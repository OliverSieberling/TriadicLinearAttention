"""Download and tokenize FineWeb-Edu into Megatron .bin/.idx (documents = BOS ... EOS, uint16).

Workers stream disjoint shards of the dataset and write shard_<i>.{bin,idx}; worker 0 first fills the val split
from the head of its stream. The shards are then concatenated into train.{bin,idx}. Deterministic for a fixed
--workers value (capped at the dataset's number of files).

    python data/prepare_fineweb.py --out-dir data/fineweb-edu --workers 24
"""

import argparse
import os
import shutil
import struct
import sys
import time
from multiprocessing import Process

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.megatron import MegatronWriter, open_megatron, read_doc_offsets, write_idx  # noqa: E402


def worker(i, args):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True).shard(num_shards=args.workers, index=i)
    shard = MegatronWriter(os.path.join(args.out_dir, f"shard_{i:03d}"))
    val = MegatronWriter(os.path.join(args.out_dir, "val")) if i == 0 else None
    writer, batch, t0 = val or shard, [], time.time()

    def flush():
        nonlocal writer
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            if ids:
                writer.add_document([tok.bos_token_id] + ids + [tok.eos_token_id])
        batch.clear()
        if writer is val and val.total_tokens >= args.val_tokens:
            val.finalize()
            writer = shard

    for ex in ds:
        batch.append(ex["text"])
        if len(batch) == args.batch_docs:
            flush()
            if i == 0 and len(shard.sizes) % (64 * args.batch_docs) < args.batch_docs and shard.total_tokens:
                print(f"[w0] {shard.total_tokens / 1e9:.2f}B shard tokens, {shard.total_tokens / (time.time() - t0) / 1e6:.1f}M tok/s", flush=True)
    flush()
    if writer is val:
        val.finalize()
    shard.finalize()
    print(f"[w{i}] shard done: {shard.total_tokens:,} tokens / {len(shard.sizes):,} docs", flush=True)


def merge(args):
    sizes = []
    with open(os.path.join(args.out_dir, "train.bin"), "wb") as out:
        for i in range(args.workers):
            p = os.path.join(args.out_dir, f"shard_{i:03d}")
            with open(p + ".idx", "rb") as f:
                f.seek(18)
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(8, 1)
                sizes.append(np.frombuffer(f.read(4 * n), dtype=np.int32))
            with open(p + ".bin", "rb") as f:
                shutil.copyfileobj(f, out, length=64 << 20)
            os.remove(p + ".bin")
            os.remove(p + ".idx")
    write_idx(os.path.join(args.out_dir, "train.idx"), np.concatenate(sizes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb_edu_100BT-shuffled")
    ap.add_argument("--subset", default=None)  # dataset config name, e.g. sample-10BT for HuggingFaceFW/fineweb-edu
    ap.add_argument("--tokenizer", default="fla-hub/transformer-1.3B-100B")  # Llama-2 32k vocabulary
    ap.add_argument("--val-tokens", type=float, default=3e8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch-docs", type=int, default=256)
    args = ap.parse_args()
    from datasets import load_dataset
    args.workers = min(args.workers, load_dataset(args.dataset, name=args.subset, split="train", streaming=True).num_shards)
    os.makedirs(args.out_dir, exist_ok=True)
    procs = [Process(target=worker, args=(i, args)) for i in range(args.workers)]
    for p in procs:
        p.start()
    for i, p in enumerate(procs):
        p.join()
        assert p.exitcode == 0, f"worker {i} failed"
    merge(args)
    for split in ("train", "val"):
        stream, offsets = open_megatron(os.path.join(args.out_dir, split)), read_doc_offsets(os.path.join(args.out_dir, split))
        assert offsets[-1] == len(stream)
        print(f"{split}: {len(stream):,} tokens / {len(offsets) - 1:,} docs")
