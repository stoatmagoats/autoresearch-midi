"""
Generate MIDI music from a trained model checkpoint.

Usage:
    uv run python generate.py                                  # random composer
    uv run python generate.py --composer chopin                 # Chopin-style
    uv run python generate.py --composer bach --n 3             # 3 Bach-style pieces
    uv run python generate.py --bars 32                         # ~32 bars long
    uv run python generate.py --bars 64 --max-tokens 8192      # longer piece
    uv run python generate.py --temperature 0.9 --top-k 50     # tweak sampling
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import sys
import time
import argparse
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    VOCAB_SIZE, COMPOSERS, MAX_SEQ_LEN,
    BOS, EOS, BAR, PAD,
    tok_comp, tok_tempo, bpm_to_bin,
    tokens_to_midi,
)

IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

# ---------------------------------------------------------------------------
# Model definition (must match train.py exactly)
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
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
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
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # KV-cache: append new K/V to cached K/V from previous steps
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)
        # When using cache, T_new=1 but K/V have full sequence — not causal masking needed
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, new_kv_cache

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        attn_out, new_kv_cache = self.attn(norm(x), ve, cos_sin, window_size, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, new_kv_cache

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
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)
        return sizes

    def forward(self, idx, targets=None, reduction='mean', kv_caches=None, start_pos=0):
        B, T = idx.size()
        # For KV-cache: cos/sin must be offset to the correct positions
        pos_end = start_pos + T
        cos_sin = self.cos[:, start_pos:pos_end], self.sin[:, start_pos:pos_end]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, ve, cos_sin, self.window_sizes[i], kv_cache=layer_cache)
            new_kv_caches.append(new_cache)
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
        return logits, new_kv_caches

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt['config'])
    model = GPT(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    vbpb = ckpt.get('val_bpb')
    print(f"Loaded model: {config.n_layer}L/{config.n_embd}D/{config.n_head}H, "
          f"vocab={config.vocab_size}, val_bpb={vbpb:.4f}, step={ckpt.get('step','?')}")
    return model

@torch.no_grad()
def generate(model, prompt_tokens, max_tokens=4096, max_bars=None,
             temperature=0.95, top_k=50, top_p=0.95,
             repetition_penalty=1.2, presence_penalty=0.3,
             ngram_penalty=2.0, ngram_sizes=(3, 4, 5, 6, 8),
             device="cuda"):
    """Generate tokens with anti-repetition measures.
    
    Anti-repetition strategy:
    1. repetition_penalty: Scale down logits for any token already in the sequence
       (standard repetition penalty, multiplicative on logits)
    2. presence_penalty: Flat additive penalty for tokens seen in last 256 tokens
    3. ngram_penalty: Penalize tokens that would complete a previously-seen n-gram
    4. bar-level detection: If last bar duplicates a recent bar, boost diversity
    """
    seq = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = list(prompt_tokens)  # CPU list for fast n-gram tracking
    bar_count = 0

    # N-gram tracking: count how many times each n-gram has appeared
    from collections import defaultdict
    ngram_counts = defaultdict(int)  # tuple -> count
    # Index: prefix -> {last_token: count} for fast lookup
    ngram_by_prefix = defaultdict(lambda: defaultdict(int))

    def _update_ngram_counts(tokens, new_idx):
        """Update n-gram counts after adding token at new_idx."""
        for n in ngram_sizes:
            if new_idx >= n - 1:
                gram = tuple(tokens[new_idx - n + 1 : new_idx + 1])
                ngram_counts[gram] += 1
                prefix = gram[:-1]
                ngram_by_prefix[(n, prefix)][gram[-1]] += 1

    def _get_bars(tokens):
        """Split token sequence into bars (list of tuples)."""
        bars = []
        current_bar = []
        for t in tokens:
            if t == BAR:
                if current_bar:
                    bars.append(tuple(current_bar))
                current_bar = []
            else:
                current_bar.append(t)
        if current_bar:
            bars.append(tuple(current_bar))
        return bars

    def _bar_repetition_streak(bars):
        """Count how many of the last N bars are identical to the most recent bar."""
        if len(bars) < 2:
            return 0
        last = bars[-1]
        streak = 0
        for b in reversed(bars[:-1]):
            if b == last:
                streak += 1
            else:
                break
        return streak

    # Structural tokens that must be allowed to repeat freely
    from prepare import is_pitch, is_dur, is_vel
    def _is_content_token(tok_id):
        """True for pitch/duration/velocity tokens — the ones we penalize for repetition."""
        return is_pitch(tok_id) or is_dur(tok_id) or is_vel(tok_id)

    # Initialize n-gram counts from prompt
    for i in range(len(generated)):
        _update_ngram_counts(generated, i)

    kv_caches = None
    # Prefill: process the entire prompt at once
    ctx = seq
    logits, kv_caches = model(ctx, kv_caches=kv_caches, start_pos=0)
    logits = logits[:, -1, :].float()
    cur_pos = seq.size(1)

    t0 = time.perf_counter()
    last_report = t0
    for step in range(max_tokens):
        # Live progress reporting
        now = time.perf_counter()
        if step > 0 and (step % 50 == 0 or now - last_report >= 2.0):
            elapsed = now - t0
            tok_s = step / elapsed if elapsed > 0 else 0
            print(f"\r    {step} tokens | {bar_count} bars | {elapsed:.1f}s | {tok_s:.1f} tok/s", end="", flush=True)
            last_report = now

        # --- Anti-repetition 1: Standard repetition penalty ---
        if repetition_penalty != 1.0:
            seen_tokens = set(generated)
            for tok_id in seen_tokens:
                if tok_id < logits.size(-1) and _is_content_token(tok_id):
                    if logits[0, tok_id] > 0:
                        logits[0, tok_id] /= repetition_penalty
                    else:
                        logits[0, tok_id] *= repetition_penalty

        # --- Anti-repetition 2: Presence penalty (recent tokens) ---
        if presence_penalty > 0:
            recent_window = generated[-256:] if len(generated) > 256 else generated
            recent_set = set(recent_window)
            for tok_id in recent_set:
                if tok_id < logits.size(-1) and _is_content_token(tok_id):
                    logits[0, tok_id] -= presence_penalty

        # --- Anti-repetition 3: N-gram penalty ---
        if ngram_penalty > 0:
            for n in ngram_sizes:
                if len(generated) >= n - 1:
                    prefix = tuple(generated[-(n - 1):])
                    token_counts = ngram_by_prefix.get((n, prefix))
                    if token_counts:
                        for tok_id, count in token_counts.items():
                            if tok_id < logits.size(-1):
                                # Escalating penalty: stronger for more repetitions and longer n-grams
                                penalty = ngram_penalty * count * (n / 4.0)
                                logits[0, tok_id] -= penalty

        # --- Anti-repetition 4: Bar-level repetition detection ---
        bars = _get_bars(generated)
        streak = _bar_repetition_streak(bars)
        if streak >= 2:
            # Bars are repeating — boost temperature to break the loop
            effective_temp = temperature * (1.0 + 0.3 * streak)
            # Also strongly penalize BAR token to discourage starting yet another copy
            logits[0, BAR] -= 2.0 * streak
        else:
            effective_temp = temperature

        # Apply temperature
        logits = logits / effective_temp

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = -float('inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = (cum - F.softmax(sorted_logits, dim=-1)) >= top_p
            sorted_logits[mask] = -float('inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        tok_val = next_tok.item()

        # Update tracking
        generated.append(tok_val)
        _update_ngram_counts(generated, len(generated) - 1)
        seq = torch.cat([seq, next_tok], dim=1)

        # Incremental forward pass: only process the new token with KV-cache
        if tok_val != EOS:
            logits, kv_caches = model(next_tok, kv_caches=kv_caches, start_pos=cur_pos)
            logits = logits[:, -1, :].float()
            cur_pos += 1

        if tok_val == EOS:
            break
        if tok_val == BAR:
            bar_count += 1
            if max_bars is not None and bar_count >= max_bars:
                eos_tok = torch.tensor([[EOS]], dtype=torch.long, device=device)
                seq = torch.cat([seq, eos_tok], dim=1)
                generated.append(EOS)
                break

    elapsed = time.perf_counter() - t0
    n_generated = len(generated) - len(prompt_tokens)
    tok_per_sec = n_generated / elapsed if elapsed > 0 else 0
    print(f"\r    Generated {n_generated} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)          ")
    return seq.squeeze(0).tolist()


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI music")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: latest from runs/)")
    parser.add_argument("--run", type=int, default=None,
                        help="Specific run number to load (e.g. --run 3)")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all available runs and exit")
    parser.add_argument("--composer", type=str, default=None,
                        help=f"Available: {', '.join(COMPOSERS)}")
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--bars", type=int, default=None,
                        help="Stop after N bars (e.g. --bars 32 for ~30s at 120 BPM)")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Multiplicative penalty for repeated tokens (1.0=off, default: 1.2)")
    parser.add_argument("--presence-penalty", type=float, default=0.3,
                        help="Additive penalty for recently seen tokens (0=off, default: 0.3)")
    parser.add_argument("--ngram-penalty", type=float, default=2.0,
                        help="Penalty for repeating n-grams (0=off, default: 2.0)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_dir, "runs")

    # --list-runs: show all available runs and exit
    if args.list_runs:
        if not os.path.isdir(runs_dir):
            print("No runs/ directory found.")
            sys.exit(0)
        import json
        run_dirs = sorted(d for d in os.listdir(runs_dir)
                          if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d)))
        print(f"{'Run':<10} {'val_bpb':>10} {'Params':>10} {'Steps':>8} {'Notes'}")
        print("-" * 70)
        for rd in run_dirs:
            cfg_path = os.path.join(runs_dir, rd, "config.json")
            has_ckpt = os.path.exists(os.path.join(runs_dir, rd, "checkpoint.pt"))
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                r = cfg.get("results", {})
                m = cfg.get("model", {})
                vbpb = f"{r.get('val_bpb', '?'):.4f}" if isinstance(r.get('val_bpb'), (int, float)) else "?"
                params = f"{m.get('num_params_M', '?')}M"
                steps = str(r.get("num_steps", "?"))
                ckpt_marker = " ✓" if has_ckpt else " (no ckpt)"
                notes = cfg.get("notes", cfg.get("description", ""))[:40]
                print(f"{rd:<10} {vbpb:>10} {params:>10} {steps:>8} {ckpt_marker} {notes}")
            else:
                print(f"{rd:<10} {'?':>10} {'?':>10} {'?':>8} (no config.json)")
        sys.exit(0)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.run is not None:
        ckpt_path = os.path.join(runs_dir, f"run_{args.run:03d}", "checkpoint.pt")
    else:
        # Auto-find latest run with a checkpoint
        ckpt_path = None
        if os.path.isdir(runs_dir):
            run_dirs = sorted(
                (d for d in os.listdir(runs_dir)
                 if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))),
                reverse=True
            )
            for rd in run_dirs:
                candidate = os.path.join(runs_dir, rd, "checkpoint.pt")
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break
        # Fallback to root checkpoint.pt
        if ckpt_path is None:
            ckpt_path = os.path.join(project_dir, "checkpoint.pt")

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(project_dir, ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, device)
    comp_map = {c: i for i, c in enumerate(COMPOSERS)}

    for idx in range(args.n):
        if args.composer:
            cname = args.composer.lower()
            if cname not in comp_map:
                print(f"Unknown composer '{cname}'. Available: {', '.join(COMPOSERS)}")
                sys.exit(1)
        else:
            cname = random.choice(COMPOSERS)
        cidx = comp_map[cname]

        prompt = [BOS, tok_comp(cidx), tok_tempo(bpm_to_bin(args.tempo))]
        bars_str = f", bars={args.bars}" if args.bars else ""
        print(f"\nGenerating {idx+1}/{args.n} (composer={cname}, tempo={args.tempo} BPM{bars_str})...")

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            tokens = generate(model, prompt, max_tokens=args.max_tokens,
                              max_bars=args.bars,
                              temperature=args.temperature, top_k=args.top_k,
                              top_p=args.top_p,
                              repetition_penalty=args.repetition_penalty,
                              presence_penalty=args.presence_penalty,
                              ngram_penalty=args.ngram_penalty,
                              device=device)

        n_bars = sum(1 for t in tokens if t == BAR)

        if args.output and args.n == 1:
            out_path = args.output
        else:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
            os.makedirs(out_dir, exist_ok=True)
            # Find next available number for this composer
            existing = [f for f in os.listdir(out_dir)
                        if f.startswith(f"{cname}_") and f.endswith(".mid")]
            nums = []
            for f in existing:
                try:
                    nums.append(int(f[len(cname)+1:-4]))
                except ValueError:
                    pass
            next_num = max(nums) + 1 if nums else 0
            out_path = os.path.join(out_dir, f"{cname}_{next_num:03d}.mid")

        tokens_to_midi(tokens, output_path=out_path)
        print(f"  → {out_path}  ({len(tokens)} tokens, {n_bars} bars)")

    print("\nDone! Open the .mid files in any MIDI player.")

if __name__ == "__main__":
    main()
