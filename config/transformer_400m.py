# 400M: 24 x 1024, 8 heads x 128, 524,288 tokens/step x 38,100 steps (20B tokens = 50 tokens/param)
# Transformer: 8 query heads per key/value head (GQA-8), QK-norm, RoPE; FFN widened to match the parameter count
mixer = 'softmax'
n_kv_head = 1
n_embd = 1024
ffn_intermediate_size = 3456
n_head = 8
max_iters = 38100
lr_decay_iters = 38100
batch_size = 4                    # per GPU
gradient_accumulation_steps = 32  # global: 2 nodes x 8 GPUs = two micro-batches per GPU
megatron_train_path = 'data/fineweb-edu/train'  # from data/prepare_fineweb.py
megatron_val_path = 'data/fineweb-edu/val'
n_layer = 24
block_size = 4096
learning_rate = 3e-4              # 1000 warmup steps, cosine decay to 3e-5
warmup_iters = 1000
wandb_run_name = 'transformer-400m'
out_dir = 'out/transformer-400m'
