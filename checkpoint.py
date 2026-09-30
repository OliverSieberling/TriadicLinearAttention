"""Checkpoints written by train.py: {'model', 'optimizer', 'model_args', 'iter_num', 'config'}."""


def load_checkpoint(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck['model'] = {k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}
    return ck


def optimizer_state_by_name(ck, gpt):
    """The checkpoint's optimizer state re-indexed for `gpt`'s optimizer.  Both index parameters as
    configure_optimizers does: decayed (2-D) parameters first, then the rest, each in model order."""
    order = lambda names: [n for n, t in names if t.dim() >= 2] + [n for n, t in names if t.dim() < 2]
    old, new = order(ck['model'].items()), order(list(gpt.named_parameters()))
    return {new.index(n): ck['optimizer']['state'][i] for i, n in enumerate(old) if i in ck['optimizer']['state']}


def load_model(path, device='cuda'):
    """The GPT of a checkpoint with its weights, on `device`."""
    from model import GPT, GPTConfig
    ck = load_checkpoint(path)
    gpt = GPT(GPTConfig(**ck['model_args']))
    gpt.load_state_dict(ck['model'])
    return gpt.to(device).eval()
