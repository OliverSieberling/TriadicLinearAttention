"""Training: DDP, document masking, deterministic data order and init.

Samples are the disjoint (block_size + 1)-token windows of the packed train stream, visited in a seeded
permutation; every batch carries the document boundaries (cu_seqlens) inside its windows.

Pretraining:  torchrun --standalone --nproc_per_node=8 train.py config/triadic_gdn_e8_400m.py
Extension:    torchrun --standalone --nproc_per_node=8 train.py config/triadic_gdn_e8_400m.py config/extend_400m.py \
                  --pretrained=out/triadic-gdn-e8-400m/ckpt_latest.pt --out_dir=out/triadic-gdn-e8-400m-64k
"""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint import load_checkpoint, optimizer_state_by_name
from data.megatron import open_megatron, read_doc_offsets, window_cu_seqlens
from model import GPT, GPTConfig

# ----------------------------------------------------------------------------
# defaults; config files and --key=value flags override them, in order (configurator.py)
out_dir = ''
init_from = 'scratch'  # 'scratch' | 'resume' (out_dir/ckpt_latest.pt) | 'pretrained' (weights of `pretrained`, fresh optimizer and schedule)
pretrained = ''        # checkpoint for init_from='pretrained' (the long-context extension starts from the 4k pretraining)
wandb_log = False
wandb_project = 'TriadicLA'
wandb_run_name = 'run'
megatron_train_path = ''
megatron_val_path = ''
vocab_size = 32000
batch_size = 8  # micro-batch per GPU
block_size = 4096
gradient_accumulation_steps = 16  # global; divided by the world size
act_ckpt = False  # recompute each block in the backward (the 64k extension)
n_layer = 24
n_embd = 1024
ffn_intermediate_size = None
mixer = 'gdn'  # 'gdn' (Triadic GDN) | 'softmax' (Transformer)
second_key_dim = 1  # E (1 = Gated DeltaNet)
n_head = 8
n_kv_head = 0
rope_theta = 10000.0
gate_rank = 128
learning_rate = 3e-4
max_iters = 38100
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-10
grad_clip = 1.0
warmup_iters = 1000  # linear warmup, then cosine to lr_decay_factor * learning_rate at lr_decay_iters
lr_decay_iters = 38100
lr_decay_factor = 0.1
eval_interval = 1000  # fixed validation set: world * eval_iters * batch_size windows
eval_iters = 16
log_interval = 10
save_checkpoint = True
milestone_frac = 0.2  # kept checkpoints ckpt_NNNNNN.pt every milestone_frac * max_iters
rolling_frac = 0.01   # ckpt_latest.pt / ckpt_prev.pt every rolling_frac * max_iters
seed = 1337  # init and data order
compile = True
dtype = 'bfloat16'
# ----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and (v is None or isinstance(v, (int, float, bool, str)))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
os.environ.setdefault('TRITON_CACHE_DIR', f'/tmp/{os.environ.get("USER", "u")}/triton-{os.getpid()}')   # JIT cache private to this process

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank, ddp_world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    torch.cuda.set_device(device)
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    ddp_rank, ddp_world_size, device = 0, 1, 'cuda'
master_process = ddp_rank == 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
assert out_dir, "set out_dir"
if master_process and save_checkpoint:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if dtype == 'bfloat16' else nullcontext()

# ----------------------------------------------------------------------------
# data
assert megatron_train_path, "set megatron_train_path"
window = block_size + 1
train_stream, train_docs = open_megatron(megatron_train_path), read_doc_offsets(megatron_train_path)
n_train_windows = len(train_stream) // window
assert n_train_windows * block_size >= max_iters * tokens_per_iter, "train split too small for a single epoch"
train_perm = np.random.default_rng(seed).permutation(n_train_windows)
train_fetches = 0
if megatron_val_path:
    val_stream, val_docs = open_megatron(megatron_val_path), read_doc_offsets(megatron_val_path)


def window_batch(stream, docs, starts):
    x = torch.stack([torch.from_numpy(np.asarray(stream[a:a + window], dtype=np.int64)) for a in starts])
    cu = torch.from_numpy(window_cu_seqlens(docs, starts, block_size))
    return (x[:, :-1].contiguous().pin_memory().to(device, non_blocking=True), x[:, 1:].contiguous().pin_memory().to(device, non_blocking=True),
            cu.pin_memory().to(device, non_blocking=True))


def get_batch():
    global train_fetches
    base = train_fetches * ddp_world_size * batch_size + ddp_rank * batch_size
    train_fetches += 1
    return window_batch(train_stream, train_docs, np.take(train_perm, np.arange(base, base + batch_size), mode='wrap') * window)


def val_batch(k):
    base = (ddp_rank * eval_iters + k) * batch_size
    return window_batch(val_stream, val_docs, np.arange(base, base + batch_size) * window)


# ----------------------------------------------------------------------------
# model
model_args = dict(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd,
                  ffn_intermediate_size=ffn_intermediate_size or math.ceil(8 * n_embd / 3 / 128) * 128,
                  mixer=mixer, second_key_dim=second_key_dim,
                  n_head=n_head, n_kv_head=n_kv_head, rope_theta=rope_theta, gate_rank=gate_rank, init_seed=seed)

iter_num, checkpoint, weights = 0, None, None
if init_from == 'scratch' and save_checkpoint and os.path.exists(os.path.join(out_dir, 'ckpt_latest.pt')):
    init_from = 'resume'  # a requeued job continues its own run instead of overwriting it
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt_latest.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = load_checkpoint(ckpt_path)
    for k, v in checkpoint['model_args'].items():
        assert model_args[k] == v, f"config mismatch on resume: {k}: {model_args[k]!r} != {v!r}"
    for k in ('seed', 'block_size', 'megatron_train_path'):
        assert checkpoint['config'].get(k) == config[k], f"resume changes {k} (would corrupt the data order)"
    assert checkpoint['config']['batch_size'] * checkpoint['config']['gradient_accumulation_steps'] == config['batch_size'] * config['gradient_accumulation_steps']
    iter_num = checkpoint['iter_num']
    train_fetches = iter_num * gradient_accumulation_steps
    weights = checkpoint['model']
elif init_from == 'pretrained':
    assert pretrained, "set pretrained=<checkpoint>"
    assert os.path.realpath(out_dir) != os.path.realpath(os.path.dirname(pretrained)), "out_dir must differ from the pretrained checkpoint's directory"
    print(f"Initialising from {pretrained} (weights only; new context length, data and schedule)")
    ck = load_checkpoint(pretrained)
    for k, v in ck['model_args'].items():   # the architecture must match; the context length and RoPE base may change
        assert k in ('block_size', 'rope_theta', 'init_seed') or model_args[k] == v, f"architecture mismatch: {k}: checkpoint {v!r} vs config {model_args[k]!r}"
    weights = ck['model']
else:
    assert init_from == 'scratch', init_from
    print("Initializing a new model from scratch")
gpt = GPT(GPTConfig(**model_args))
if weights is not None:
    gpt.load_state_dict(weights)
gpt.act_ckpt = act_ckpt
gpt.to(device)
optimizer = gpt.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), eps)
if checkpoint is not None:
    sd = optimizer.state_dict()
    sd['state'] = optimizer_state_by_name(checkpoint, gpt)
    optimizer.load_state_dict(sd)
checkpoint = weights = None
model = torch.compile(gpt) if compile else gpt
if ddp:
    model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        X, Y, CU = val_batch(k)
        pos_ids, block_mask = gpt.prepare_doc_batch(CU, batch_size * block_size)
        with ctx:
            losses[k] = model(X, Y, cu_seqlens=CU, pos_ids=pos_ids, block_mask=block_mask)[1]
    loss = losses.mean()
    if ddp:
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
    model.train()
    return loss.item()


def get_lr(it):
    if it <= warmup_iters:
        return learning_rate * it / warmup_iters
    if it >= lr_decay_iters:
        return learning_rate * lr_decay_factor
    cos = (1.0 + math.cos(math.pi * (it - warmup_iters) / (lr_decay_iters - warmup_iters))) / 2
    return learning_rate * ((1.0 - lr_decay_factor) * cos + lr_decay_factor)


def save_ckpt(path):
    torch.save({'model': gpt.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': asdict(gpt.config),
                'iter_num': iter_num, 'config': config}, path + '.tmp')
    os.replace(path + '.tmp', path)


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=os.environ.get('WANDB_NAME', wandb_run_name), config=config)

# ----------------------------------------------------------------------------
# training loop
milestone_interval = max(1, round(max_iters * milestone_frac))
rolling_interval = max(1, round(max_iters * rolling_frac))
X, Y, CU = get_batch()
pos_ids, block_mask = gpt.prepare_doc_batch(CU, batch_size * block_size)
while iter_num < max_iters:
    lr = get_lr(iter_num)
    for group in optimizer.param_groups:
        group['lr'] = lr
    if megatron_val_path and iter_num % eval_interval == 0:
        val_loss = estimate_loss()
        if master_process:
            print(f"step {iter_num}: val/loss {val_loss:.4f}")
            if wandb_log:
                wandb.log({"iter": iter_num, "val/loss": val_loss}, step=iter_num)

    t0 = time.time()
    loss_sum = None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss = model(X, Y, cu_seqlens=CU, pos_ids=pos_ids, block_mask=block_mask)
        loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        X, Y, CU = get_batch()  # prefetch while the GPU is busy
        pos_ids, block_mask = gpt.prepare_doc_batch(CU, batch_size * block_size)
        (loss / gradient_accumulation_steps).backward()
    if ddp:
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.AVG)
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    if iter_num % log_interval == 0 and master_process:
        lossf = (loss_sum / gradient_accumulation_steps).item()
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, peak mem {mem:.1f}GiB")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": lossf, "lr": lr, "step_time_ms": dt * 1000,
                       "tokens": iter_num * tokens_per_iter, "peak_mem_gb": mem}, step=iter_num)
    iter_num += 1
    if save_checkpoint and master_process:
        if iter_num % rolling_interval == 0 or iter_num == max_iters:
            latest, prev = os.path.join(out_dir, 'ckpt_latest.pt'), os.path.join(out_dir, 'ckpt_prev.pt')
            save_ckpt(os.path.join(out_dir, 'ckpt_new.pt'))
            if os.path.exists(latest):
                os.replace(latest, prev)
            os.replace(os.path.join(out_dir, 'ckpt_new.pt'), latest)
        if iter_num % milestone_interval == 0:
            save_ckpt(os.path.join(out_dir, f'ckpt_{iter_num:06d}.pt'))

if ddp:
    destroy_process_group()
