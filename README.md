# Triadic Linear Attention

Minimal training codebase for
[Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling](https://arxiv.org/abs/2609.36529),
based on [nanoGPT](https://github.com/karpathy/nanoGPT). The paper applies the triadic construction to linear
attention in general. This repository contains its main instance, Triadic Gated DeltaNet (Triadic GDN), and the
Transformer baseline. Triadic GDN keeps `E` state slices per head. The state is written with the outer product of
a key, a second key and a value, and read with two queries. `E = 1` is Gated DeltaNet. The kernels are in
[cute-triadic-gdn](https://github.com/OliverSieberling/cute-triadic-gdn).

```
model.py          GPT with two sequence mixers: Triadic GDN and softmax attention
train.py          pretraining and long-context extension (DDP, document masking, checkpoints)
checkpoint.py     loading checkpoints
config/           one file per model and scale, plus extend_400m.py and extend_1.3b.py for the 64k stage
data/             Megatron .bin/.idx reader and writer, FineWeb-Edu, PG19 and extension-mix preparation
```

## Install

Python 3.12 and H100 GPUs. The kernels are built for sm90 and compile at first use.

```
pip install torch==2.12.1 nvidia-cutlass-dsl==4.7.1 datasets transformers zstandard wandb
pip install git+https://github.com/OliverSieberling/cute-triadic-gdn
```

## Data

Pretraining data (FineWeb-Edu with the Llama-2 32k tokenizer, about 100B tokens and 200 GB):

```
python data/prepare_fineweb.py --out-dir data/fineweb-edu --workers 24
```

Extension mix (50 % FineWeb-Edu, 10 % PG19, 40 % scientific PDFs, 2.1B tokens at 400M and 7.2B at 1.3B):

```
python data/prepare_pg19.py --out-dir data/pg19 --train-tokens 210e6
python data/prepare_extension_mix.py --stage pdf --out-dir data/extension-400m --tokens 2.1e9
python data/prepare_extension_mix.py --stage mix --out-dir data/extension-400m --tokens 2.1e9
```

For the 1.3B models use `--train-tokens 720e6`, `--tokens 7.2e9` and `data/extension-1.3b`.

## Training

Pretraining at 4k context, then the 64k extension from the pretrained checkpoint:

```
torchrun --standalone --nproc_per_node=8 train.py config/triadic_gdn_e8_400m.py
torchrun --standalone --nproc_per_node=8 train.py config/triadic_gdn_e8_400m.py config/extend_400m.py \
    --pretrained=out/triadic-gdn-e8-400m/ckpt_latest.pt --out_dir=out/triadic-gdn-e8-400m-64k
```

The configs are `gdn`, `triadic_gdn_e2`, `triadic_gdn_e4`, `triadic_gdn_e8` and `transformer`, each at `400m` and
`1.3b`, with the paper's hyperparameters. `gradient_accumulation_steps` is global, so the tokens per step do not
depend on the number of GPUs. For several nodes, run one `torchrun --nnodes=N --node_rank=... --rdzv_endpoint=...`
per node. Any config value can be overridden on the command line, for example `--second_key_dim=16 --seed=1440`.
Checkpoints go to `out_dir`. A run started in an `out_dir` that already holds `ckpt_latest.pt` resumes from it.

## Checkpoints

```python
from checkpoint import load_model
gpt = load_model('out/triadic-gdn-e8-400m-64k/ckpt_latest.pt')
logits, _ = gpt(idx)          # idx: [B, T] token ids, one document per row
_, loss = gpt(idx, targets)   # document boundaries within a row via cu_seqlens
```

## License

MIT.

## Citation

```bibtex
@misc{sieberling2026triadic,
  title        = {Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling},
  author       = {Sieberling, Oliver and Runwal, Bharat and Jin, David and Chin, Ryan and Panda, Rameswar and Kim, Yoon},
  year         = {2026},
  eprint       = {2609.36529},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2609.36529}
}
```
