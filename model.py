"""GPT with two sequence mixers: Triadic Gated DeltaNet (second-key dimension E; E = 1 is Gated DeltaNet)
and softmax attention (the Transformer baseline).

Every batch is one packed row of B*T tokens with document boundaries cu_seqlens: the recurrences reset there,
softmax attention is causal within a document, RoPE positions restart per document.
"""

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from cute_triadic_gdn import conv_split_act_call, gdn_joint_call

HEAD_DIM = 128
CONV_SIZE = 4
NORM_EPS = 1e-5
INIT_STD = 0.02


@dataclass
class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 24
    n_embd: int = 1024
    ffn_intermediate_size: int = None  # None -> 8/3 * n_embd rounded up to a multiple of 128
    mixer: str = 'gdn'                 # 'gdn' (Triadic GDN, E = second_key_dim) | 'softmax'
    second_key_dim: int = 1            # E: 1 = Gated DeltaNet, >1 = Triadic GDN with E state slices per head
    n_head: int = 8                    # n_head * HEAD_DIM == n_embd
    n_kv_head: int = 0                 # softmax: GQA key/value heads (0 = n_head)
    rope_theta: float = 10000.0        # softmax: RoPE base (10k for pretraining, 2M for the 64k extension)
    gate_rank: int = 128               # gdn: rank of the sigmoid output gate
    init_seed: int = 1337


def l2norm(t):
    tf = t.float()
    return (tf * torch.rsqrt((tf * tf).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)


class GatedRMSNorm(nn.Module):
    """RMSNorm of x with a learned scale, multiplied by sigmoid(g); computed in fp32, returned in x's dtype."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x, g):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + NORM_EPS) * self.weight.float() * torch.sigmoid(g.float())
        return y.to(x.dtype)


def cross_entropy(x, targets, weight, chunk=2048):
    """Mean cross-entropy over the tokens whose target is not -1, from hidden states x [N, C] and the output projection
    weight [V, C]; the logits of each chunk of tokens are recomputed in the backward instead of being kept."""
    def chunk_loss(xc, yc):
        return F.cross_entropy(F.linear(xc, weight).float(), yc, ignore_index=-1, reduction='sum')
    total = sum(torch.utils.checkpoint.checkpoint(chunk_loss, x[s:s + chunk], targets[s:s + chunk], use_reentrant=False)
                for s in range(0, x.shape[0], chunk))
    return total / (targets != -1).sum().clamp(min=1)


class CausalConv(nn.Module):
    """Depthwise causal convolution (width CONV_SIZE) over `channels`, SiLU on the first `act_channels` and none on
    the rest, in one kernel."""

    def __init__(self, channels, act_channels):
        super().__init__()
        self.act_channels = act_channels
        self.weight = nn.Parameter(torch.empty(channels, CONV_SIZE))

    def forward(self, x, cu_seqlens):
        return conv_split_act_call(x, self.weight, self.act_channels, cu_seqlens)


_FLEX = None


def _flex():
    # eager flex_attention materialises dense scores; outside a compiled forward use a compiled instance
    global _FLEX
    if torch.compiler.is_compiling():
        return flex_attention
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=True)
    return _FLEX


class Attention(nn.Module):
    """Softmax attention: GQA, QK-norm, RoPE, causal within documents (FlexAttention block mask)."""

    def __init__(self, cfg):
        super().__init__()
        H, HK, D = cfg.n_head, cfg.n_kv_head or cfg.n_head, HEAD_DIM
        self.n_head, self.n_kv_head = H, HK
        self.in_proj = nn.Linear(cfg.n_embd, (H + 2 * HK) * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.n_embd, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=NORM_EPS)
        self.k_norm = nn.RMSNorm(D, eps=NORM_EPS)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = torch.outer(torch.arange(cfg.block_size, dtype=torch.float32), inv_freq)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)

    def _rope(self, x, pos):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c, s = self.rope_cos[pos][None, None].to(x.dtype), self.rope_sin[pos][None, None].to(x.dtype)
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)

    def forward(self, x, cu_seqlens, pos_ids, block_mask):
        B, T, C = x.shape
        D = HEAD_DIM
        q, k, v = self.in_proj(x).split([self.n_head * D, self.n_kv_head * D, self.n_kv_head * D], dim=-1)
        q = self.q_norm(q.view(B, T, self.n_head, D)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.n_kv_head, D)).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, D).transpose(1, 2)
        q, k = self._rope(q, pos_ids), self._rope(k, pos_ids)
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        y = _flex()(q, k, v, block_mask=block_mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, C))


class TriadicGDN(nn.Module):
    """Triadic Gated DeltaNet (paper Algorithm 2): per head a third-order state of E slices S_e (D x D each), written
    with the outer product of the key, the second key k2 (E scalars: short conv, softplus, L2-normalised over E) and the
    value, and read with the query and the second query q2.  Each slice has its own decay gate alpha_e; the delta rule
    erases under both keys jointly with the write strength beta.  E = 1 is Gated DeltaNet (k2 = q2 = 1, no second
    projections).

        S_e <- alpha_te S_e;  S <- S + beta_t (k2_t (x) k_t) (v_t - S^T (k2_t (x) k_t))^T;  o_t = S^T (q2_t (x) q_t) / sqrt(D)

    q, k, v, k2, q2 come from one GEMM and one causal depthwise convolution (SiLU on q, k, v only); the decay gates,
    the write strength and the rank-`gate_rank` output-gate input from a second GEMM.  The output passes a
    sigmoid-gated RMSNorm."""

    def __init__(self, cfg):
        super().__init__()
        H, D, E = cfg.n_head, HEAD_DIM, cfg.second_key_dim
        self.n_head, self.E = H, E
        self.qkv_sizes = [H * D] * 3 + [H * E] * 2 * (E > 1)      # q, k, v, then k2 and q2 when E > 1
        self.gate_sizes = [H * E, H, cfg.gate_rank]                # decay gates, write strength, output-gate input
        self.qkv_proj = nn.Linear(cfg.n_embd, sum(self.qkv_sizes), bias=False)
        self.qkv_conv1d = CausalConv(sum(self.qkv_sizes), act_channels=3 * H * D)
        self.gate_proj = nn.Linear(cfg.n_embd, sum(self.gate_sizes), bias=False)
        self.A_log = nn.Parameter(torch.empty(H * E))
        self.decay_bias = nn.Parameter(torch.empty(H * E))
        self.gate_out = nn.Linear(cfg.gate_rank, H * D, bias=True)
        self.o_norm = GatedRMSNorm(D)
        self.o_proj = nn.Linear(H * D, cfg.n_embd, bias=False)

    def forward(self, x, cu_seqlens, pos_ids=None, block_mask=None):
        B, T, _ = x.shape
        H, D, E = self.n_head, HEAD_DIM, self.E
        q, k, v, *second = self.qkv_conv1d(self.qkv_proj(x), cu_seqlens).split(self.qkv_sizes, dim=-1)
        decay_in, beta_in, gate_in = self.gate_proj(x).split(self.gate_sizes, dim=-1)
        q, k, v = (t.reshape(B, T, H, D) for t in (q, k, v))
        if E > 1:
            k2, q2 = (l2norm(F.softplus(t).reshape(B, T, H, E).float()) for t in second)
        else:
            k2 = q2 = torch.ones(B, T, H, 1, device=x.device, dtype=torch.float32)
        log_alpha = -(self.A_log.float().exp().view(H, E) * F.softplus(decay_in.float().view(B, T, H, E) + self.decay_bias.float().view(H, E)))
        o = gdn_joint_call(l2norm(q).to(torch.bfloat16), l2norm(k).to(torch.bfloat16), v.to(torch.bfloat16),
                           k2, q2, log_alpha, beta_in.sigmoid().float(), scale=D ** -0.5, cu_seqlens=cu_seqlens)
        o = self.o_norm(o, self.gate_out(gate_in).view(B, T, H, D))
        return self.o_proj(o.reshape(B, T, H * D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.n_embd, 2 * cfg.ffn_intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.RMSNorm(cfg.n_embd, eps=NORM_EPS)
        self.attn = Attention(cfg) if cfg.mixer == 'softmax' else TriadicGDN(cfg)
        self.ln_2 = nn.RMSNorm(cfg.n_embd, eps=NORM_EPS)
        self.mlp = MLP(cfg)

    def forward(self, x, cu_seqlens, pos_ids, block_mask):
        x = x + self.attn(self.ln_1(x), cu_seqlens, pos_ids, block_mask)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.ffn_intermediate_size is None:
            cfg.ffn_intermediate_size = math.ceil(8 * cfg.n_embd / 3 / 128) * 128
        assert cfg.n_head * HEAD_DIM == cfg.n_embd
        assert cfg.mixer in ('gdn', 'softmax')
        self.config = cfg
        self.act_ckpt = False   # runtime switch (train.py): recompute each block in the backward
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.RMSNorm(cfg.n_embd, eps=NORM_EPS),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.init_weights()
        print(f"number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    @torch.no_grad()
    def init_weights(self):
        # per-parameter seeded init (independent of construction order): weights N(0, 0.02), conv U(+-1/sqrt(width)),
        # A_log ~ log U(0, 16), decay_bias = softplus^-1(dt) with dt ~ LogU(1e-3, 1e-1), biases 0, norms 1
        for name, p in self.named_parameters():
            g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"{self.config.init_seed}:{name}".encode()).digest()[:8], 'little') % 2**63)
            t = torch.empty(p.shape, dtype=torch.float32)
            if name.endswith('A_log'):
                t.uniform_(0.0, 16.0, generator=g).log_()
            elif name.endswith('decay_bias'):
                dt = torch.exp(t.uniform_(0.0, 1.0, generator=g) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
                t = dt + torch.log(-torch.expm1(-dt))
            elif 'conv1d' in name:
                t.uniform_(-1.0 / math.sqrt(CONV_SIZE), 1.0 / math.sqrt(CONV_SIZE), generator=g)
            elif name.endswith('bias'):
                t.zero_()
            elif p.dim() >= 2:
                t.normal_(0.0, INIT_STD, generator=g)
            else:
                t.fill_(1.0)
            p.copy_(t)

    @torch.no_grad()
    def prepare_doc_batch(self, cu_seqlens, total_tokens=None):
        """Document-relative positions [total] and, for softmax attention, the causal same-document block mask.
        `total_tokens` (= batch x block_size) saves the device->host read of cu_seqlens[-1]."""
        device = cu_seqlens.device
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        total = total_tokens if total_tokens is not None else int(cu_seqlens[-1].item())
        doc_start = cu_seqlens[:-1].long().repeat_interleave(lengths, output_size=total)
        pos_ids = torch.arange(total, device=device) - doc_start
        if self.config.mixer != 'softmax':
            return pos_ids, None
        BS = 128
        nb = (total + BS - 1) // BS
        doc_ids = torch.full((nb * BS,), -1, device=device, dtype=torch.int32)  # -1 pads the last block
        doc_ids[:total] = torch.arange(len(lengths), device=device, dtype=torch.int32).repeat_interleave(lengths)

        def doc_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc_ids[kv_idx] >= 0) & (doc_ids[q_idx] == doc_ids[kv_idx])

        qb = torch.arange(nb, device=device)
        lo = doc_start[qb * BS] // BS
        kv_idx = (lo[:, None] + qb[None, :]).clamp_(max=nb - 1).to(torch.int32)
        block_mask = BlockMask.from_kv_blocks((qb - lo + 1).to(torch.int32)[None, None], kv_idx[None, None],
                                              BLOCK_SIZE=BS, mask_mod=doc_causal, seq_lengths=(total, total))
        return pos_ids, block_mask

    def _hidden(self, idx, cu_seqlens, pos_ids, block_mask):
        b, t = idx.size()
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, b + 1, device=idx.device, dtype=torch.int32) * t
        if pos_ids is None:
            pos_ids, block_mask = self.prepare_doc_batch(cu_seqlens)
        x = self.transformer.wte(idx.view(1, b * t))
        for block in self.transformer.h:
            if self.act_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cu_seqlens, pos_ids, block_mask, use_reentrant=False)
            else:
                x = block(x, cu_seqlens, pos_ids, block_mask)
        return self.transformer.ln_f(x)

    def forward(self, idx, targets=None, cu_seqlens=None, pos_ids=None, block_mask=None):
        """idx: [B, T]; the batch is flattened to one packed row described by cu_seqlens (default: each row one document).
        With targets, returns (None, mean loss); without, the logits of the last position of each row."""
        b, t = idx.size()
        x = self._hidden(idx, cu_seqlens, pos_ids, block_mask)
        if targets is not None:
            return None, cross_entropy(x[0], targets.reshape(-1), self.lm_head.weight)
        return self.lm_head(x.view(b, t, -1)[:, [-1], :]), None

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps):
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
        print(f"decayed params: {sum(p.numel() for p in decay):,}; non-decayed: {sum(p.numel() for p in no_decay):,}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps, fused=True)
