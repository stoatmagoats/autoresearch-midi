"""
MIDI music generation training script. Single-GPU, single-file.
Adapted from autoresearch pretraining for REMI-style MIDI token prediction.
Usage: uv run train.py
"""

import os
import json
import signal
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
# Persist inductor cache (survives reboots, unlike /tmp/torchinductor_*)
_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".inductor_cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"  # cache FX graph for faster warmup
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(os.cpu_count())  # parallel kernel compilation

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Detect AMD ROCm vs NVIDIA CUDA
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

# Flash Attention 3 (CUDA-only via kernels package) is not used on ROCm.
# ROCm uses PyTorch SDPA which dispatches to AOTriton.
fa3 = None

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if IS_ROCM:
            # PyTorch SDPA on ROCm dispatches to AOTriton
            # Note: SDPA doesn't support window_size, so SSSL pattern degrades to
            # full causal attention on all layers
            q = q.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Optimizer fused steps: compiled on CUDA, eager on ROCm.
# The model-level torch.compile (applied later with monkey-patch) is the main speedup.
_maybe_compile = torch.compile(dynamic=False, fullgraph=True) if not IS_ROCM else lambda fn: fn

@_maybe_compile
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    dtype = exp_avg.dtype
    exp_avg.lerp_(grad, (1 - beta1_t).to(dtype=dtype))
    exp_avg_sq.lerp_(grad.square(), (1 - beta2_t).to(dtype=dtype))
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@_maybe_compile
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), (1 - beta2).to(dtype=second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters — tuned for MIDI music generation
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 48       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # smaller head dim for more heads at moderate width
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 64 * 2048  # 131072 tokens per optimizer step (1 fwd pass per step)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 12              # number of transformer layers
DEVICE_BATCH_SIZE = 64   # per-device batch size (uses ~52GB VRAM, more steps/hr)

# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

def _next_run_dir():
    """Find the next available run_NNN directory."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(RUNS_DIR)
                if d.startswith("run_") and os.path.isdir(os.path.join(RUNS_DIR, d))]
    if not existing:
        num = 1
    else:
        nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
        num = max(nums) + 1 if nums else 1
    run_dir = os.path.join(RUNS_DIR, f"run_{num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, num

RUN_DIR, RUN_NUM = _next_run_dir()
print(f"Run directory: {RUN_DIR}")

# Check for checkpoint resumption
RESUME_RUN = os.environ.get("RESUME_RUN")
RESUME_CKPT = None
if RESUME_RUN:
    resume_dir = os.path.join(RUNS_DIR, f"run_{int(RESUME_RUN):03d}")
    resume_ckpt_path = os.path.join(resume_dir, "checkpoint.pt")
    if os.path.exists(resume_ckpt_path):
        RESUME_CKPT = resume_ckpt_path
        print(f"Will resume from: {RESUME_CKPT}")
    else:
        print(f"WARNING: No checkpoint found at {resume_ckpt_path}, starting fresh")

def _get_hyperparams():
    """Capture all hyperparameters as a dict for config.json."""
    return {
        "TIME_BUDGET": TIME_BUDGET,
        "DEPTH": DEPTH,
        "ASPECT_RATIO": ASPECT_RATIO,
        "HEAD_DIM": HEAD_DIM,
        "DEVICE_BATCH_SIZE": DEVICE_BATCH_SIZE,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "MATRIX_LR": MATRIX_LR,
        "EMBEDDING_LR": EMBEDDING_LR,
        "UNEMBEDDING_LR": UNEMBEDDING_LR,
        "SCALAR_LR": SCALAR_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "ADAM_BETAS": list(ADAM_BETAS),
        "WARMUP_RATIO": WARMUP_RATIO,
        "WARMDOWN_RATIO": WARMDOWN_RATIO,
        "FINAL_LR_FRAC": FINAL_LR_FRAC,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "WINDOW_PATTERN": WINDOW_PATTERN,
    }

# Set up logging to run directory
import sys as _sys
_log_path = os.path.join(RUN_DIR, "run.log")
_log_file = open(_log_path, "w")

class _Tee:
    """Write to both stdout and log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_sys.stdout = _Tee(_sys.__stdout__, _log_file)
_sys.stderr = _Tee(_sys.__stderr__, _log_file)
print(f"Logging to {_log_path}")

# Write initial config.json (results will be updated after training)
_initial_config = {
    "run_id": f"run_{RUN_NUM:03d}",
    "status": "running",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "resumed_from": f"run_{int(RESUME_RUN):03d}" if RESUME_RUN else None,
    "hyperparameters": _get_hyperparams(),
}
with open(os.path.join(RUN_DIR, "config.json"), "w") as _f:
    json.dump(_initial_config, _f, indent=2)
print(f"Initial config written to {RUN_DIR}/config.json")

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# Peak BF16 FLOPS by GPU model (for MFU calculation)
_GPU_PEAK_FLOPS = {
    "H100":   989.5e12,
    "H200":   989.5e12,
    "A100":   312.0e12,
    "B200":   2250.0e12,
    # AMD Instinct
    "MI300X": 1307.4e12,
    "MI308X": 1307.4e12,
    "MI325X": 1307.4e12,
    "MI250X": 383.0e12,
    # AMD Radeon (RDNA)
    "8060S":  139.3e12,
}

def _detect_peak_flops():
    gpu_name = torch.cuda.get_device_name(0)
    for key, flops in _GPU_PEAK_FLOPS.items():
        if key.lower() in gpu_name.lower():
            print(f"Detected GPU: {gpu_name} -> peak BF16 FLOPS: {flops:.1e}")
            return flops
    print(f"Warning: Unknown GPU '{gpu_name}', defaulting to H100 peak FLOPS for MFU")
    return 989.5e12

PEAK_BF16_FLOPS = _detect_peak_flops()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile: fix ROCm inductor benchmarking bug, then enable on all platforms.
# Bug: InductorBenchmarker.benchmark_gpu crashes with ZeroDivisionError when
# estimated_timing is 0.0 (kernel too fast for CUDA event resolution on gfx1151).
# Fix: guard the division. Benchmarked at 1.21× speedup on our 12-layer model.
if IS_ROCM:
    import torch._inductor.runtime.benchmarking as _benchmarking
    from torch._inductor.runtime.benchmarking import time_and_count as _tc

    def _patched_benchmark_gpu(self, _callable, estimation_iters=5, memory_warmup_iters=100,
                               benchmark_iters=100, max_benchmark_duration=25,
                               return_mode="min", grad_to_none=None, **kwargs):
        torch.cuda.synchronize()
        _callable()
        torch.cuda.synchronize()
        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()
        event_pairs = self.get_event_pairs(estimation_iters)
        for s, e in event_pairs:
            if grad_to_none:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_(); s.record(); _callable(); e.record()
        torch.cuda.synchronize()
        estimated_timing = self.get_event_pairs_min_timing(event_pairs)
        if estimated_timing > 0:
            benchmark_iters = max(
                min(benchmark_iters, int(max_benchmark_duration // estimated_timing)), 1
            )
        for _ in range(memory_warmup_iters):
            buffer.zero_()
        event_pairs = self.get_event_pairs(benchmark_iters)
        for s, e in event_pairs:
            if grad_to_none:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_(); s.record(); _callable(); e.record()
        torch.cuda.synchronize()
        del buffer
        if return_mode == "all":
            return [s.elapsed_time(e) for s, e in event_pairs]
        elif return_mode == "min":
            bt = self.get_event_pairs_min_timing(event_pairs)
            return min(estimated_timing, bt) if estimated_timing > 0 else bt
        else:
            raise ValueError(f"Unsupported return_mode: {return_mode}")

    _benchmarking.InductorBenchmarker.benchmark_gpu = _tc(_patched_benchmark_gpu)
    print("ROCm: patched InductorBenchmarker (ZeroDivisionError fix)")

model = torch.compile(model, dynamic=False)
print("torch.compile: enabled")

# Resume from checkpoint if requested
resumed_training_time = 0.0
resumed_step = 0
resumed_smooth_loss = 0.0
if RESUME_CKPT:
    print(f"Loading checkpoint from {RESUME_CKPT}...")
    ckpt = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"  Resumed model + optimizer from step {ckpt.get('step', '?')}, val_bpb={ckpt.get('val_bpb', '?')}")
    else:
        print(f"  Resumed model (no optimizer state) from step {ckpt.get('step', '?')}, val_bpb={ckpt.get('val_bpb', '?')}")
    # Restore training state for pause/resume
    resumed_training_time = ckpt.get('total_training_time', 0.0)
    resumed_step = ckpt.get('step', 0)
    resumed_smooth_loss = ckpt.get('smooth_train_loss', 0.0)
    if resumed_training_time > 0:
        remaining = max(0, TIME_BUDGET - resumed_training_time)
        print(f"  Resuming with {resumed_training_time:.0f}s already trained, {remaining:.0f}s remaining")
    del ckpt

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# SIGINT handler: Ctrl+C pauses training gracefully
_pause_requested = False
def _sigint_handler(signum, frame):
    global _pause_requested
    if _pause_requested:
        print("\nForce quit (second Ctrl+C)")
        exit(1)
    _pause_requested = True
    print("\n⏸  Pause requested — finishing current step, then saving checkpoint...")
signal.signal(signal.SIGINT, _sigint_handler)

t_start_training = time.time()
smooth_train_loss = resumed_smooth_loss
total_training_time = resumed_training_time
step = resumed_step

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_BF16_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > resumed_step + 10 and total_training_time >= TIME_BUDGET:
        break

    # Pause requested via Ctrl+C
    if _pause_requested:
        break

paused = _pause_requested
if paused:
    print("\n⏸  Training paused.")
else:
    print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval (skip if paused — save time, we'll eval on resume completion)
if not paused:
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
    print(f"val_bpb: {val_bpb:.6f}")
else:
    val_bpb = None
    print("Skipping val eval (paused — will eval on resume completion)")

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
effective_steps = step - max(resumed_step, 0)
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * max(effective_steps - 10, 1) / max(total_training_time - resumed_training_time, 1) / PEAK_BF16_FLOPS
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# Save checkpoint (includes training state for pause/resume)
checkpoint_path = os.path.join(RUN_DIR, "checkpoint.pt")
checkpoint_data = {
    'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': asdict(config),
    'vocab_size': vocab_size,
    'step': step,
    'total_training_time': total_training_time,
    'smooth_train_loss': smooth_train_loss,
}
if val_bpb is not None:
    checkpoint_data['val_bpb'] = val_bpb
torch.save(checkpoint_data, checkpoint_path)

# Symlink checkpoint.pt in project root → latest run
root_ckpt = os.path.join(PROJECT_DIR, "checkpoint.pt")
try:
    if os.path.islink(root_ckpt) or os.path.exists(root_ckpt):
        os.remove(root_ckpt)
    os.symlink(checkpoint_path, root_ckpt)
except OSError:
    pass  # symlink may fail on some filesystems
print(f"Checkpoint saved to {checkpoint_path}")

# Save config.json with hyperparameters and results
run_status = "paused" if paused else "completed"
results = {
    "val_bpb": val_bpb,
    "training_seconds": round(total_training_time, 1),
    "total_seconds": round(t_end - t_start, 1),
    "peak_vram_mb": round(peak_vram_mb, 1),
    "mfu_percent": round(steady_state_mfu, 2),
    "total_tokens_M": round(total_tokens / 1e6, 1),
    "num_steps": step,
    "epochs": epoch,
    "status": run_status,
}
run_config = {
    "run_id": f"run_{RUN_NUM:03d}",
    "status": run_status,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "resumed_from": f"run_{int(RESUME_RUN):03d}" if RESUME_RUN else None,
    "hyperparameters": _get_hyperparams(),
    "model": {
        "model_dim": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "num_params_M": round(num_params / 1e6, 1),
    },
    "dataset": {
        "composers": len(tokenizer.composers),
        "vocab_size": vocab_size,
    },
    "results": results,
}
config_path = os.path.join(RUN_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)
print(f"Config saved to {config_path}")

print("---")
if val_bpb is not None:
    print(f"val_bpb:          {val_bpb:.6f}")
else:
    print(f"val_bpb:          (skipped — paused)")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
if paused:
    print(f"\n💡 Resume this run with: RESUME_RUN={RUN_NUM} uv run python train.py")
