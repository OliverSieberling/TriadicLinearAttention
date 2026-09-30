"""The long-context extension mix: 50 % FineWeb-Edu replay, 10 % PG19 books, 40 % scientific PDFs (OLMo 3's
s2pdf long-context sources), documents kept whole and globally shuffled, as Megatron .bin/.idx.

    python data/prepare_extension_mix.py --stage pdf --out-dir data/extension-400m --tokens 2.1e9
    python data/prepare_extension_mix.py --stage mix --out-dir data/extension-400m --tokens 2.1e9 \
        --fineweb data/fineweb-edu/train --pg19 data/pg19/train

`pdf` streams the three s2pdf sources of allenai/dolma3_longmino_mix-50B-1025 (documents of 8-16k, 16-32k and
32-64k tokens in OLMo's tokens, at OLMo's proportions among them) into <out-dir>/pdf.{bin,idx}; `mix` assembles
<out-dir>/train.{bin,idx} from that, the PG19 training shard (data/prepare_pg19.py) and random FineWeb-Edu documents.
--tokens is the mix size: 2.1e9 for the 400M models, 7.2e9 for the 1.3B models (about 5 tokens per parameter)."""

import argparse
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.megatron import MegatronWriter, open_megatron, read_doc_offsets  # noqa: E402

REPO = "allenai/dolma3_longmino_mix-50B-1025"
# s2pdf sources of the OLMo long-context mix and their share of the PDF part (OLMo's proportions, renormalised)
PDF_SOURCES = {r"^data/olmocr_science_pdfs-high_quality.*-2e15$": 0.539,   # 32-64k tokens
               r"^data/olmocr_science_pdfs-high_quality.*-2e13$": 0.254,   # 8-16k
               r"^data/olmocr_science_pdfs-high_quality.*-2e14$": 0.207}   # 16-32k
SHARES = dict(fineweb=0.50, pg19=0.10, pdf=0.40)


def stage_pdf(args):
    import zstandard
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    api = HfApi()
    w = MegatronWriter(os.path.join(args.out_dir, "pdf"))
    for pattern, share in PDF_SOURCES.items():
        target = args.tokens * SHARES["pdf"] * share
        dirs = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data") if re.match(pattern, f.path))
        files = [f.path for d in dirs for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=d) if f.path.endswith(".jsonl.zst")]
        np.random.default_rng(args.seed).shuffle(files)
        start, batch = w.total_tokens, []
        print(f"{pattern}: {len(files)} files, target {target / 1e6:.0f}M tokens", flush=True)

        def flush():
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                w.add_document([bos] + ids + [eos])
            batch.clear()

        for fp in files:
            if w.total_tokens - start >= target:
                break
            local = hf_hub_download(REPO, fp, repo_type="dataset")
            with open(local, "rb") as fh:
                for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                    batch.append(json.loads(line)["text"])
                    if len(batch) == 8:
                        flush()
                    if w.total_tokens - start >= target:
                        break
            flush()
            os.remove(local)
            print(f"  {w.total_tokens - start:,} tokens after {os.path.basename(fp)}", flush=True)
    w.finalize()
    print(f"pdf: {len(w.sizes):,} documents, {w.total_tokens:,} tokens")


def stage_mix(args):
    rng = np.random.default_rng(args.seed)
    entries = []
    sources = {}
    for name, path in (("fineweb", args.fineweb), ("pg19", args.pg19), ("pdf", os.path.join(args.out_dir, "pdf"))):
        stream, offsets = open_megatron(path), read_doc_offsets(path)
        sources[name] = (stream, offsets)
        budget, total = args.tokens * SHARES[name], 0
        for d in rng.permutation(len(offsets) - 1):
            if total >= budget:
                break
            entries.append((name, int(d)))
            total += int(offsets[d + 1] - offsets[d])
        assert total >= budget * 0.99, f"{name}: only {total:,} of {budget:,.0f} tokens available"
    rng.shuffle(entries)
    w = MegatronWriter(os.path.join(args.out_dir, "train"))
    composition = {}
    for i, (name, d) in enumerate(entries):
        stream, offsets = sources[name]
        doc = np.asarray(stream[offsets[d]:offsets[d + 1]], dtype=np.uint16)
        composition[name] = composition.get(name, 0) + len(doc)
        w.add_document(doc)
        if i % 50000 == 0:
            print(f"mix: {i:,}/{len(entries):,} documents, {w.total_tokens:,} tokens", flush=True)
    w.finalize()
    json.dump(dict(tokens=w.total_tokens, documents=len(w.sizes), composition=composition, shares=SHARES),
              open(os.path.join(args.out_dir, "train.meta.json"), "w"), indent=1)
    print(f"train: {len(w.sizes):,} documents, {w.total_tokens:,} tokens; composition " +
          ", ".join(f"{k} {v / w.total_tokens:.1%}" for k, v in composition.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pdf", "mix"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", type=float, required=True, help="size of the mix: 2.1e9 (400M) or 7.2e9 (1.3B)")
    ap.add_argument("--fineweb", default="data/fineweb-edu/train")
    ap.add_argument("--pg19", default="data/pg19/train")
    ap.add_argument("--tokenizer", default="fla-hub/transformer-1.3B-100B")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    (stage_pdf if args.stage == "pdf" else stage_mix)(args)
