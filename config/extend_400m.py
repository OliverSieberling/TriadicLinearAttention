# 64k extension of a 400M pretraining, stacked after the model config (pass --pretrained and --out_dir):
# 524,288 tokens/step (8 GPUs x 1 window of 65,536) x 3,815 steps = 2.0B tokens (5 tokens/param) of the extension mix,
# learning rate 1e-4 with 10 % warmup and cosine decay to 1e-5; blocks recomputed in the backward to fit 64k;
# Transformers raise the RoPE base to 2M.
init_from = 'pretrained'
megatron_train_path = 'data/extension-400m/train'  # from data/prepare_extension_mix.py
megatron_val_path = ''
block_size = 65536
batch_size = 1
gradient_accumulation_steps = 8
max_iters = 3815
lr_decay_iters = 3815
warmup_iters = 381
learning_rate = 1e-4
rope_theta = 2000000.0
act_ckpt = True
