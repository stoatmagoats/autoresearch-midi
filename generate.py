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

# ---------------------------------------------------------------------------
# Motif-Aware Repetition Control (Experiment 1.3)
# ---------------------------------------------------------------------------

from collections import defaultdict
from prepare import is_pitch, is_dur, is_vel, dec_pitch, dec_vel, tok_vel, NUM_VEL_BINS

class MotifAwareRepetitionControl:
    """Hierarchical repetition controller that allows musical motif return
    while preventing degenerate loops.

    Key concepts:
    - Bar fingerprint: hash of pitch classes in a bar (ignoring velocity/timing)
    - Consecutive streak: count of identical adjacent bars (the degenerate case)
    - Motif return: a bar similar to one heard 2-8 bars ago (the musical case)
    - Penalty scale: multiplier on all penalties — low normally, high during loops

    When consecutive_streak < 2:   penalty_scale = 0.3  (allow natural repetition)
    When consecutive_streak == 2:  penalty_scale = 2.0  (getting suspicious)
    When consecutive_streak >= 3:  penalty_scale = 3.0  (stuck in a loop)
    """

    def __init__(self, motif_return_bonus=0.4, motif_return_window=(2, 8)):
        self.bar_tokens = []          # tokens in the current (incomplete) bar
        self.bar_fingerprints = []    # list of pitch-class tuples, one per completed bar
        self.bar_pitch_sets = []      # list of sets of MIDI pitches per completed bar
        self.consecutive_streak = 0
        self.motif_return_bonus = motif_return_bonus
        self.motif_min, self.motif_max = motif_return_window
        # Track which pitch fingerprints appeared at which bar indices
        self.fingerprint_to_bars = defaultdict(list)  # fingerprint -> [bar_idx, ...]

    def on_token(self, tok_val):
        """Call after each generated token to update internal state."""
        if tok_val == BAR:
            self._complete_bar()
            self.bar_tokens = []
        else:
            self.bar_tokens.append(tok_val)

    def _complete_bar(self):
        """Called when a BAR token is generated — finalize the previous bar."""
        if not self.bar_tokens:
            return

        # Extract pitch classes from the bar (ignoring position, duration, velocity)
        pitches = tuple(sorted(
            dec_pitch(t) % 12 for t in self.bar_tokens if is_pitch(t)
        ))
        midi_pitches = set(
            dec_pitch(t) for t in self.bar_tokens if is_pitch(t)
        )

        bar_idx = len(self.bar_fingerprints)
        self.bar_fingerprints.append(pitches)
        self.bar_pitch_sets.append(midi_pitches)
        self.fingerprint_to_bars[pitches].append(bar_idx)

        # Track consecutive identical bars
        if bar_idx > 0 and self.bar_fingerprints[bar_idx - 1] == pitches:
            self.consecutive_streak += 1
        else:
            self.consecutive_streak = 0

    def get_penalty_scale(self):
        """Returns a multiplier for all anti-repetition penalties.
        < 1.0 = relaxed (allow repetition), > 1.0 = tightened (prevent loops).
        """
        if self.consecutive_streak >= 3:
            return 3.0    # strongly penalize — stuck in a loop
        elif self.consecutive_streak >= 2:
            return 2.0    # getting suspicious — escalate
        else:
            return 0.3    # allow natural musical repetition

    def get_motif_return_bonus_pitches(self):
        """Returns a set of MIDI pitch token IDs that should get a small logit
        boost because they belong to a bar heard 2-8 bars ago (motif return).

        Only activates when not in a loop (consecutive_streak < 2).
        Returns dict: {pitch_token_id: bonus_amount}
        """
        if self.consecutive_streak >= 2:
            return {}  # don't boost during loops

        n_bars = len(self.bar_fingerprints)
        if n_bars < self.motif_min:
            return {}

        # Collect pitches from bars that appeared motif_min..motif_max bars ago
        bonus_pitches = {}
        lo = max(0, n_bars - self.motif_max)
        hi = max(0, n_bars - self.motif_min + 1)

        for bar_idx in range(lo, hi):
            for midi_pitch in self.bar_pitch_sets[bar_idx]:
                from prepare import tok_pitch
                tok_id = tok_pitch(midi_pitch)
                # Slightly boost — the bonus decreases with distance
                distance = n_bars - bar_idx
                amt = self.motif_return_bonus * (1.0 - (distance - self.motif_min) /
                       max(1, self.motif_max - self.motif_min))
                bonus_pitches[tok_id] = max(bonus_pitches.get(tok_id, 0), amt)

        return bonus_pitches

    def get_loop_temperature_multiplier(self):
        """Extra temperature multiplier when stuck in a loop."""
        if self.consecutive_streak >= 2:
            return 1.0 + 0.3 * self.consecutive_streak
        return 1.0

    def should_penalize_bar_token(self):
        """Returns a penalty to apply to the BAR token when in a loop."""
        if self.consecutive_streak >= 2:
            return 2.0 * self.consecutive_streak
        return 0.0


# ---------------------------------------------------------------------------
# Dynamic Arc Controller — velocity momentum for coherent builds/releases
# ---------------------------------------------------------------------------

class DynamicArcController:
    """Tracks velocity trends across bars and nudges generation toward coherent
    dynamic arcs (builds → climaxes, releases → quiet sections).

    Problem: The model generates good local dynamics but can abruptly switch
    from a build to a quiet section because its 2048-token context window
    can't see the full arc.

    Solution: Track the velocity trend over the last N bars. If velocity is
    rising (build), gently penalize sudden drops. If velocity is falling
    (release), gently penalize sudden spikes. Allow gradual transitions.

    The bias is soft — the model CAN change direction, but only gradually
    over 2-3 bars rather than instantly.
    """

    def __init__(self, momentum_strength=1.5, trend_window=4, max_jump=10):
        self.bar_avg_velocities = []  # average velocity bin per completed bar
        self.current_bar_vels = []    # velocity bins in current (incomplete) bar
        self.momentum_strength = momentum_strength
        self.trend_window = trend_window
        self.max_jump = max_jump  # max allowed velocity bin jump per bar

    def on_token(self, tok_val):
        """Call after each generated token."""
        if is_vel(tok_val):
            self.current_bar_vels.append(dec_vel(tok_val))
        elif tok_val == BAR:
            if self.current_bar_vels:
                avg = sum(self.current_bar_vels) / len(self.current_bar_vels)
                self.bar_avg_velocities.append(avg)
            self.current_bar_vels = []

    def get_velocity_bias(self):
        """Returns dict of {vel_token_id: logit_bias} to apply.
        Positive bias = boost, negative = penalize.
        """
        if len(self.bar_avg_velocities) < 2 or self.momentum_strength <= 0:
            return {}

        window = self.bar_avg_velocities[-self.trend_window:]
        if len(window) < 2:
            return {}

        # Compute linear trend: positive = crescendo, negative = diminuendo
        trend = (window[-1] - window[0]) / len(window)
        recent_avg = window[-1]  # use most recent bar, not window average

        biases = {}
        for vel_bin in range(NUM_VEL_BINS):
            tok_id = tok_vel(vel_bin)
            distance_from_recent = vel_bin - recent_avg

            if abs(trend) > 0.5:  # meaningful trend detected
                if trend > 0:  # BUILDING — penalize sudden drops
                    if distance_from_recent < -self.max_jump:
                        # Big drop during a build — penalize proportionally
                        overshoot = abs(distance_from_recent) - self.max_jump
                        biases[tok_id] = -self.momentum_strength * (overshoot / NUM_VEL_BINS)
                    elif distance_from_recent > 0:
                        # Continuing the build — small boost
                        biases[tok_id] = self.momentum_strength * 0.15
                else:  # RELEASING — penalize sudden spikes
                    if distance_from_recent > self.max_jump:
                        # Big spike during a release — penalize proportionally
                        overshoot = distance_from_recent - self.max_jump
                        biases[tok_id] = -self.momentum_strength * (overshoot / NUM_VEL_BINS)
                    elif distance_from_recent < 0:
                        # Continuing the release — small boost
                        biases[tok_id] = self.momentum_strength * 0.15
            else:
                # No strong trend — gently penalize extreme jumps in either direction
                if abs(distance_from_recent) > self.max_jump:
                    overshoot = abs(distance_from_recent) - self.max_jump
                    biases[tok_id] = -self.momentum_strength * 0.3 * (overshoot / NUM_VEL_BINS)

        return biases

    @property
    def trend_str(self):
        """Human-readable trend for progress display."""
        if len(self.bar_avg_velocities) < 2:
            return ""
        window = self.bar_avg_velocities[-self.trend_window:]
        if len(window) < 2:
            return ""
        trend = (window[-1] - window[0]) / len(window)
        if trend > 1.5:
            return "↑build"
        elif trend > 0.5:
            return "↗rise"
        elif trend < -1.5:
            return "↓release"
        elif trend < -0.5:
            return "↘fade"
        return "→steady"


@torch.no_grad()
def generate(model, prompt_tokens, max_tokens=4096, max_bars=None,
             temperature=0.95, top_k=50, top_p=0.95,
             repetition_penalty=1.2, presence_penalty=0.3,
             ngram_penalty=2.0, ngram_sizes=(3, 4, 5, 6, 8),
             repetition_mode="smart", motif_return_bonus=0.4,
             dynamic_momentum=1.5,
             device="cuda"):
    """Generate tokens with motif-aware anti-repetition.

    repetition_mode:
        'smart'      — Motif-aware: low penalties normally, high during loops,
                       with motif return bonus (NEW DEFAULT)
        'aggressive' — Original flat penalties (old behavior)
        'off'        — No anti-repetition penalties at all
    """
    seq = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = list(prompt_tokens)  # CPU list for fast n-gram tracking
    bar_count = 0

    # N-gram tracking (used in both smart and aggressive modes)
    ngram_counts = defaultdict(int)
    ngram_by_prefix = defaultdict(lambda: defaultdict(int))

    def _update_ngram_counts(tokens, new_idx):
        """Update n-gram counts after adding token at new_idx."""
        for n in ngram_sizes:
            if new_idx >= n - 1:
                gram = tuple(tokens[new_idx - n + 1 : new_idx + 1])
                ngram_counts[gram] += 1
                prefix = gram[:-1]
                ngram_by_prefix[(n, prefix)][gram[-1]] += 1

    def _is_content_token(tok_id):
        """True for pitch/duration/velocity tokens — the ones we penalize."""
        return is_pitch(tok_id) or is_dur(tok_id) or is_vel(tok_id)

    # Initialize n-gram counts from prompt
    for i in range(len(generated)):
        _update_ngram_counts(generated, i)

    # Initialize motif controller (used in 'smart' mode)
    motif_ctrl = MotifAwareRepetitionControl(
        motif_return_bonus=motif_return_bonus,
    )
    # Initialize dynamic arc controller
    arc_ctrl = DynamicArcController(
        momentum_strength=dynamic_momentum,
    )
    # Feed prompt tokens into controllers
    for t in prompt_tokens:
        motif_ctrl.on_token(t)
        arc_ctrl.on_token(t)

    kv_caches = None
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
            extra = ""
            if repetition_mode == "smart":
                arc_info = arc_ctrl.trend_str
                extra = f" {arc_info}" if arc_info else ""
            print(f"\r    {step} tokens | {bar_count} bars | {elapsed:.1f}s | {tok_s:.1f} tok/s{extra}", end="", flush=True)
            last_report = now

        effective_temp = temperature

        if repetition_mode == "smart":
            # --- SMART MODE: motif-aware penalties ---
            scale = motif_ctrl.get_penalty_scale()

            # Scaled repetition penalty (gentle normally, strong during loops)
            if repetition_penalty != 1.0:
                # Effective penalty: lerp between 1.0 (no penalty) and full penalty
                eff_rep = 1.0 + (repetition_penalty - 1.0) * scale
                seen_tokens = set(generated[-512:] if len(generated) > 512 else generated)
                for tok_id in seen_tokens:
                    if tok_id < logits.size(-1) and _is_content_token(tok_id):
                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= eff_rep
                        else:
                            logits[0, tok_id] *= eff_rep

            # Scaled presence penalty
            if presence_penalty > 0:
                eff_pres = presence_penalty * scale
                recent_window = generated[-256:] if len(generated) > 256 else generated
                recent_set = set(recent_window)
                for tok_id in recent_set:
                    if tok_id < logits.size(-1) and _is_content_token(tok_id):
                        logits[0, tok_id] -= eff_pres

            # Scaled n-gram penalty
            if ngram_penalty > 0:
                eff_ngram = ngram_penalty * scale
                for n in ngram_sizes:
                    if len(generated) >= n - 1:
                        prefix = tuple(generated[-(n - 1):])
                        token_counts = ngram_by_prefix.get((n, prefix))
                        if token_counts:
                            for tok_id, count in token_counts.items():
                                if tok_id < logits.size(-1):
                                    penalty = eff_ngram * count * (n / 4.0)
                                    logits[0, tok_id] -= penalty

            # Motif return bonus: boost pitches from bars heard 2-8 bars ago
            bonus_pitches = motif_ctrl.get_motif_return_bonus_pitches()
            for tok_id, bonus in bonus_pitches.items():
                if tok_id < logits.size(-1):
                    logits[0, tok_id] += bonus

            # Dynamic arc guidance: velocity momentum
            if dynamic_momentum > 0:
                vel_biases = arc_ctrl.get_velocity_bias()
                for tok_id, bias in vel_biases.items():
                    if tok_id < logits.size(-1):
                        logits[0, tok_id] += bias

            # Loop-breaking: temperature boost + BAR penalty
            effective_temp *= motif_ctrl.get_loop_temperature_multiplier()
            bar_penalty = motif_ctrl.should_penalize_bar_token()
            if bar_penalty > 0:
                logits[0, BAR] -= bar_penalty

        elif repetition_mode == "aggressive":
            # --- AGGRESSIVE MODE: original flat penalties (old behavior) ---
            if repetition_penalty != 1.0:
                seen_tokens = set(generated)
                for tok_id in seen_tokens:
                    if tok_id < logits.size(-1) and _is_content_token(tok_id):
                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= repetition_penalty
                        else:
                            logits[0, tok_id] *= repetition_penalty

            if presence_penalty > 0:
                recent_window = generated[-256:] if len(generated) > 256 else generated
                recent_set = set(recent_window)
                for tok_id in recent_set:
                    if tok_id < logits.size(-1) and _is_content_token(tok_id):
                        logits[0, tok_id] -= presence_penalty

            if ngram_penalty > 0:
                for n in ngram_sizes:
                    if len(generated) >= n - 1:
                        prefix = tuple(generated[-(n - 1):])
                        token_counts = ngram_by_prefix.get((n, prefix))
                        if token_counts:
                            for tok_id, count in token_counts.items():
                                if tok_id < logits.size(-1):
                                    penalty = ngram_penalty * count * (n / 4.0)
                                    logits[0, tok_id] -= penalty

            # Bar-level streak detection (old behavior)
            def _get_bars_from(tokens):
                bars, current = [], []
                for t in tokens:
                    if t == BAR:
                        if current: bars.append(tuple(current))
                        current = []
                    else: current.append(t)
                if current: bars.append(tuple(current))
                return bars

            bars = _get_bars_from(generated)
            if len(bars) >= 2:
                last = bars[-1]
                streak = sum(1 for b in reversed(bars[:-1]) if b == last)
                if streak >= 2:
                    effective_temp = temperature * (1.0 + 0.3 * streak)
                    logits[0, BAR] -= 2.0 * streak

        # 'off' mode: no penalties applied

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
        if repetition_mode == "smart":
            motif_ctrl.on_token(tok_val)
            arc_ctrl.on_token(tok_val)
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
    mode_str = f" [{repetition_mode}]" if repetition_mode != "smart" else ""
    print(f"\r    Generated {n_generated} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s){mode_str}          ")
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
    parser.add_argument("--repetition-mode", type=str, default="smart",
                        choices=["smart", "aggressive", "off"],
                        help="Repetition control mode: smart (motif-aware, default), "
                             "aggressive (old flat penalties), off (no penalties)")
    parser.add_argument("--motif-bonus", type=float, default=0.4,
                        help="Logit bonus for pitches from bars heard 2-8 bars ago (default: 0.4)")
    parser.add_argument("--dynamic-momentum", type=float, default=1.5,
                        help="Velocity momentum strength for coherent builds/releases (0=off, default: 1.5)")
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

    print(f"Repetition mode: {args.repetition_mode}")

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
                              repetition_mode=args.repetition_mode,
                              motif_return_bonus=args.motif_bonus,
                              dynamic_momentum=args.dynamic_momentum,
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

