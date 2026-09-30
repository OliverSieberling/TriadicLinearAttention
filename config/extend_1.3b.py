# 64k extension of a 1.3B pretraining, stacked after the model config (pass --pretrained and --out_dir):
# 1,048,576 tokens/step (16 GPUs x 1 window of 65,536) x 6,500 steps = 6.8B tokens (5 tokens/param) of the extension mix,
# learning rate 1e-4 with 10 % warmup and cosine decay to 1e-5; blocks recomputed in the backward to fit 64k.
init_from = 'pretrained'
megatron_train_path = 'data/extension-1.3b/train'
megatron_val_path = ''
block_size = 65536
batch_size = 1
gradient_accumulation_steps = 16
max_iters = 6500
lr_decay_iters = 6500
warmup_iters = 650
learning_rate = 1e-4
rope_theta = 2000000.0
act_ckpt = True
