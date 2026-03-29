"""
MIDI data preparation for music generation experiments.
Tokenizes MIDI files from midi_files/ using REMI-style encoding.

Usage:
    python prepare.py          # tokenize all MIDI files and cache
    python prepare.py --stats  # show dataset statistics
"""

import os
import sys
import math
import json
import random
import argparse
import multiprocessing
from pathlib import Path

import mido
import torch

# ---------------------------------------------------------------------------
# Constants (shared with train.py)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 8192
TIME_BUDGET = 57600        # 16 hours for MIDI training (10× augmented dataset)
MIDI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midi_files")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".midi_cache")

# Quantization
STEPS_PER_BEAT = 4         # 16th-note resolution
MAX_POS_PER_BAR = 32       # supports up to 8/4 time
MAX_DUR_STEPS = 64         # max note duration in 16th notes
NUM_VEL_BINS = 32
NUM_TEMPO_BINS = 32
TEMPO_MIN, TEMPO_MAX = 40, 240  # BPM range

# Data split
VAL_RATIO = 0.1
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Token Vocabulary
# ---------------------------------------------------------------------------

# Chord vocabulary: 12 roots × 5 qualities = 60 tokens + 1 "no chord"
CHORD_QUALITIES = ['maj', 'min', 'dim', 'aug', 'dom7']
NUM_CHORD_TOKENS = 12 * len(CHORD_QUALITIES) + 1  # 61

PAD, BOS, EOS, BAR = 0, 1, 2, 3
NUM_SPECIAL = 4

POS_OFF   = NUM_SPECIAL                        # 4
PITCH_OFF = POS_OFF + MAX_POS_PER_BAR           # 36
DUR_OFF   = PITCH_OFF + 128                     # 164
VEL_OFF   = DUR_OFF + MAX_DUR_STEPS             # 228
TEMPO_OFF = VEL_OFF + NUM_VEL_BINS              # 260
COMP_OFF  = TEMPO_OFF + NUM_TEMPO_BINS           # 292

def _get_composers():
    if not os.path.isdir(MIDI_DIR):
        return []
    return sorted(d for d in os.listdir(MIDI_DIR)
                  if os.path.isdir(os.path.join(MIDI_DIR, d)))

COMPOSERS = _get_composers()
CHORD_OFF = COMP_OFF + len(COMPOSERS)
VOCAB_SIZE = CHORD_OFF + NUM_CHORD_TOKENS

# --- Encoding helpers ---
def tok_pos(p):   return POS_OFF   + min(max(p, 0), MAX_POS_PER_BAR - 1)
def tok_pitch(p): return PITCH_OFF + min(max(p, 0), 127)
def tok_dur(d):   return DUR_OFF   + min(max(d, 1), MAX_DUR_STEPS) - 1
def tok_vel(v):   return VEL_OFF   + min(max(v, 0), NUM_VEL_BINS - 1)
def tok_tempo(t): return TEMPO_OFF + min(max(t, 0), NUM_TEMPO_BINS - 1)
def tok_comp(i):  return COMP_OFF  + i
def tok_chord(root, qual_idx): return CHORD_OFF + root * len(CHORD_QUALITIES) + qual_idx
def tok_chord_none():          return CHORD_OFF + 60

# --- Decoding helpers ---
def dec_pos(t):   return t - POS_OFF
def dec_pitch(t): return t - PITCH_OFF
def dec_dur(t):   return (t - DUR_OFF) + 1        # 1-indexed
def dec_vel(t):   return t - VEL_OFF
def dec_tempo(t): return t - TEMPO_OFF
def dec_comp(t):  return t - COMP_OFF
def dec_chord(t): return t - CHORD_OFF

def is_pos(t):   return POS_OFF   <= t < POS_OFF   + MAX_POS_PER_BAR
def is_pitch(t): return PITCH_OFF <= t < PITCH_OFF + 128
def is_dur(t):   return DUR_OFF   <= t < DUR_OFF   + MAX_DUR_STEPS
def is_vel(t):   return VEL_OFF   <= t < VEL_OFF   + NUM_VEL_BINS
def is_tempo(t): return TEMPO_OFF <= t < TEMPO_OFF + NUM_TEMPO_BINS
def is_comp(t):  return COMP_OFF  <= t < COMP_OFF  + len(COMPOSERS)
def is_chord(t): return CHORD_OFF <= t < CHORD_OFF + NUM_CHORD_TOKENS

# --- Velocity / tempo quantization ---
def vel_to_bin(v):   return min(v * NUM_VEL_BINS // 128, NUM_VEL_BINS - 1)
def bin_to_vel(b):   return int((b + 0.5) * 128 / NUM_VEL_BINS)
def bpm_to_bin(bpm):
    bpm = max(TEMPO_MIN, min(TEMPO_MAX, bpm))
    return min(int((bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN) * NUM_TEMPO_BINS),
               NUM_TEMPO_BINS - 1)
def bin_to_bpm(b):
    return int(TEMPO_MIN + (b + 0.5) / NUM_TEMPO_BINS * (TEMPO_MAX - TEMPO_MIN))

# ---------------------------------------------------------------------------
# MIDI Parsing → tokens
# ---------------------------------------------------------------------------

def detect_chord(pitches):
    """Given a list of MIDI pitches, detect the most likely chord.
    Returns (root_pc, quality_idx) or None."""
    if len(pitches) < 2:
        return None
    # Pitch class histogram
    pc_hist = [0] * 12
    for p in pitches:
        pc_hist[p % 12] += 1
    # Template matching against chord templates
    templates = {
        'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # root, M3, P5
        'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # root, m3, P5
        'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # root, m3, d5
        'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # root, M3, A5
        'dom7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], # root, M3, P5, m7
    }
    best_score, best_root, best_qual = -1, 0, 0
    for root in range(12):
        for qi, (qname, template) in enumerate(templates.items()):
            # Rotate template to this root
            rotated = template[-root:] + template[:-root]
            score = sum(a * b for a, b in zip(pc_hist, rotated))
            if score > best_score:
                best_score = score
                best_root = root
                best_qual = qi
    return (best_root, best_qual) if best_score >= 2 else None

# Piano range for transposition filtering

# Piano range for transposition filtering
PIANO_MIN, PIANO_MAX = 21, 108  # A0 to C8


def _parse_midi(filepath):
    """Parse MIDI → list of (start_tick, pitch, dur_ticks, velocity), metadata."""
    mid = mido.MidiFile(filepath)
    tpb = mid.ticks_per_beat
    tempo_us = 500000       # default 120 BPM
    ts_num, ts_den = 4, 4

    notes = []
    for track in mid.tracks:
        t = 0
        active = {}
        for msg in track:
            t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (t, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    s, v = active.pop(msg.note)
                    if t - s > 0:
                        notes.append((s, msg.note, t - s, v))

    notes.sort(key=lambda x: (x[0], x[1]))
    bpm = 60_000_000 / tempo_us
    return notes, tpb, bpm, ts_num, ts_den


def transpose_notes(notes, semitones):
    """Transpose all notes by N semitones. Returns None if any note goes out of piano range."""
    transposed = []
    for start, pitch, dur, vel in notes:
        new_pitch = pitch + semitones
        if new_pitch < PIANO_MIN or new_pitch > PIANO_MAX:
            return None  # skip this transposition entirely
        transposed.append((start, new_pitch, dur, vel))
    return transposed


def _tokenize_from_notes(notes, tpb, bpm, ts_num, ts_den, composer_idx):
    """Quantize parsed notes and build REMI token sequence.

    This is the shared core used by both tokenize_file() (original) and
    the transposition augmentation path.
    """
    if not notes:
        return []

    ticks_per_step = tpb // STEPS_PER_BEAT
    beats_per_bar = ts_num * (4 / ts_den)
    steps_per_bar = int(beats_per_bar * STEPS_PER_BEAT)

    # Quantize
    qnotes = []
    for st, pit, dur, vel in notes:
        s = round(st / ticks_per_step)
        d = max(1, min(round(dur / ticks_per_step), MAX_DUR_STEPS))
        bar = s // steps_per_bar
        pos = s % steps_per_bar
        qnotes.append((bar, pos, pit, d, vel_to_bin(vel)))
    qnotes.sort(key=lambda x: (x[0], x[1], x[2]))

    # Build tokens
    from collections import defaultdict
    bars = defaultdict(list)
    for bar, pos, pit, dur, vb in qnotes:
        bars[bar].append((pos, pit, dur, vb))

    tokens = [BOS, tok_comp(composer_idx), tok_tempo(bpm_to_bin(bpm))]
    for bar_num in sorted(bars.keys()):
        tokens.append(BAR)
        
        # Detect chord for this bar
        bar_pitches = [pit for _, pit, _, _ in bars[bar_num]]
        chord = detect_chord(bar_pitches)
        if chord:
            root, qual = chord
            tokens.append(tok_chord(root, qual))
        else:
            tokens.append(tok_chord_none())
            
        # Add notes
        for pos, pit, dur, vb in sorted(bars[bar_num]):
            tokens.extend([tok_pos(pos), tok_pitch(pit), tok_dur(dur), tok_vel(vb)])
    tokens.append(EOS)
    return tokens


def tokenize_file(filepath, composer_idx):
    """Tokenize a single MIDI file → list[int]."""
    notes, tpb, bpm, ts_num, ts_den = _parse_midi(filepath)
    return _tokenize_from_notes(notes, tpb, bpm, ts_num, ts_den, composer_idx)

# ---------------------------------------------------------------------------
# Tokens → MIDI
# ---------------------------------------------------------------------------

def tokens_to_midi(tokens, output_path=None, default_bpm=120):
    """Convert token list back to a playable MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tps = 480 // STEPS_PER_BEAT           # ticks per step (120)
    steps_per_bar = 4 * STEPS_PER_BEAT    # assume 4/4

    bpm = default_bpm
    notes = []
    cur_bar = -1
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in (BOS, EOS, PAD) or is_comp(t) or is_chord(t):
            i += 1; continue
        if is_tempo(t):
            bpm = bin_to_bpm(dec_tempo(t)); i += 1; continue
        if t == BAR:
            cur_bar += 1; i += 1; continue
        if is_pos(t) and i + 3 < len(tokens):
            p, pi, di, vi = t, tokens[i+1], tokens[i+2], tokens[i+3]
            if is_pitch(pi) and is_dur(di) and is_vel(vi):
                bar = max(0, cur_bar)
                abs_step = bar * steps_per_bar + dec_pos(p)
                notes.append((abs_step * tps, dec_pitch(pi),
                              dec_dur(di) * tps, bin_to_vel(dec_vel(vi))))
                i += 4; continue
        i += 1

    # Build MIDI messages
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

    events = []
    for st, pit, dur, vel in notes:
        events.append((st, 1, pit, vel))       # note_on
        events.append((st + dur, 0, pit, 0))   # note_off
    events.sort(key=lambda x: (x[0], x[1]))

    last = 0
    for abs_t, is_on, pit, vel in events:
        delta = abs_t - last
        if is_on:
            track.append(mido.Message('note_on', note=pit, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', note=pit, velocity=0, time=delta))
        last = abs_t
    track.append(mido.MetaMessage('end_of_track', time=0))

    if output_path:
        mid.save(output_path)
    return mid

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal wrapper compatible with train.py interface."""
    def __init__(self):
        self._vocab_size = VOCAB_SIZE
        self.composers = COMPOSERS
        self.composer_to_idx = {c: i for i, c in enumerate(COMPOSERS)}

    @classmethod
    def from_directory(cls):
        return cls()

    def get_vocab_size(self):
        return self._vocab_size

    def get_bos_token_id(self):
        return BOS


def make_dataloader(tokenizer, B, T, split):
    """Yield (inputs, targets, epoch) from cached token tensors."""
    assert split in ("train", "val")
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise RuntimeError(f"Cache not found: {path}. Run prepare.py first.")

    data = torch.load(path, map_location="cpu", weights_only=True)
    n = len(data)
    row_len = T + 1

    # Pre-allocate pinned CPU + GPU buffers
    cpu_buf = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buf = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_x = cpu_buf[:B * T].view(B, T)
    cpu_y = cpu_buf[B * T:].view(B, T)
    x = gpu_buf[:B * T].view(B, T)
    y = gpu_buf[B * T:].view(B, T)

    epoch = 1
    pos = 0
    while True:
        for r in range(B):
            if pos + row_len > n:
                pos = 0
                epoch += 1
            cpu_x[r] = data[pos:pos + T]
            cpu_y[r] = data[pos + 1:pos + row_len]
            pos += T          # stride by T (non-overlapping)
        gpu_buf.copy_(cpu_buf, non_blocking=True)
        yield x, y, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits-per-token on validation set (reported as val_bpb for compat)."""
    path = os.path.join(CACHE_DIR, "val.pt")
    val_data = torch.load(path, map_location="cpu", weights_only=True)
    n_val = len(val_data)
    tpb = batch_size * MAX_SEQ_LEN
    steps = max(1, n_val // tpb)

    loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_loss, total_count = 0.0, 0
    for _ in range(steps):
        xi, yi, _ = next(loader)
        mask = (yi != PAD).float()
        loss_flat = model(xi, yi, reduction='none').view(-1)
        m = mask.view(-1)
        total_loss += (loss_flat * m).sum().item()
        total_count += m.sum().item()

    if total_count == 0:
        return float('inf')
    return total_loss / (math.log(2) * total_count)

# ---------------------------------------------------------------------------
# Main — tokenize MIDI files and cache
# ---------------------------------------------------------------------------

# Transposition augmentation range: -5 to +6 semitones (skip 0 = original)
TRANSPOSE_RANGE = range(-5, 7)  # [-5, -4, ..., -1, 0, 1, ..., 6]


def _process_one_file(args):
    """Process a single MIDI file: tokenize original + all transpositions.

    Designed to run in a multiprocessing.Pool worker.
    Returns (sequences, skipped, n_originals, n_transposed, error_msg).
    """
    fp, comp, composer_idx_map = args
    cidx = composer_idx_map[comp]
    sequences = []
    n_originals = 0
    n_transposed = 0

    try:
        notes, tpb, bpm, ts_num, ts_den = _parse_midi(fp)

        # Original
        toks = _tokenize_from_notes(notes, tpb, bpm, ts_num, ts_den, cidx)
        if len(toks) > 10:
            sequences.append(toks)
            n_originals = 1
        else:
            return sequences, 1, 0, 0, None  # skipped

        # Transpositions
        for shift in TRANSPOSE_RANGE:
            if shift == 0:
                continue
            t_notes = transpose_notes(notes, shift)
            if t_notes is not None:
                toks_t = _tokenize_from_notes(t_notes, tpb, bpm, ts_num, ts_den, cidx)
                if len(toks_t) > 10:
                    sequences.append(toks_t)
                    n_transposed += 1

    except Exception as e:
        return [], 1, 0, 0, f"  Skipped {fp}: {e}"

    return sequences, 0, n_originals, n_transposed, None


def _augment_with_transpositions(parsed_entries, composer_idx_map):
    """Generate original + transposed token sequences from parsed MIDI entries.

    Uses multiprocessing to parallelize across CPU cores.

    Args:
        parsed_entries: list of (filepath, composer_name) tuples
        composer_idx_map: dict mapping composer name → index

    Returns:
        (sequences, skipped, n_originals, n_transposed)
        where sequences is a list of token lists
    """
    work_items = [(fp, comp, composer_idx_map) for fp, comp in parsed_entries]
    n_workers = min(multiprocessing.cpu_count(), len(work_items))

    sequences = []
    skipped = 0
    n_originals = 0
    n_transposed = 0

    with multiprocessing.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_process_one_file, work_items, chunksize=16):
            seqs, sk, no, nt, err = result
            sequences.extend(seqs)
            skipped += sk
            n_originals += no
            n_transposed += nt
            if err:
                print(err)

    return sequences, skipped, n_originals, n_transposed


def prepare_data():
    os.makedirs(CACHE_DIR, exist_ok=True)

    comp_map = {c: i for i, c in enumerate(COMPOSERS)}
    entries = []
    for comp in COMPOSERS:
        cdir = os.path.join(MIDI_DIR, comp)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(('.mid', '.midi')):
                entries.append((os.path.join(cdir, fn), comp))

    print(f"Found {len(entries)} MIDI files from {len(COMPOSERS)} composers")
    print(f"Transposition augmentation: shifts {list(TRANSPOSE_RANGE)} (skip 0)")

    # -----------------------------------------------------------------------
    # Split ORIGINAL files into train/val FIRST, then augment within each split
    # This prevents data leakage (a val piece's transposition in training).
    # -----------------------------------------------------------------------
    rng = random.Random(SPLIT_SEED)
    idx = list(range(len(entries)))
    rng.shuffle(idx)
    n_val = max(1, int(len(entries) * VAL_RATIO))
    val_indices = set(idx[:n_val])

    train_entries = [entries[i] for i in range(len(entries)) if i not in val_indices]
    val_entries   = [entries[i] for i in range(len(entries)) if i in val_indices]
    print(f"Split: {len(train_entries)} train files, {len(val_entries)} val files (before augmentation)")

    # Augment each split independently
    print("Tokenizing + augmenting train split...")
    train_seqs, train_skip, train_orig, train_trans = _augment_with_transpositions(train_entries, comp_map)
    print(f"  Train: {train_orig} originals + {train_trans} transpositions = {len(train_seqs)} sequences ({train_skip} skipped)")

    print("Tokenizing + augmenting val split...")
    val_seqs, val_skip, val_orig, val_trans = _augment_with_transpositions(val_entries, comp_map)
    print(f"  Val:   {val_orig} originals + {val_trans} transpositions = {len(val_seqs)} sequences ({val_skip} skipped)")

    total_seqs = len(train_seqs) + len(val_seqs)
    total_skip = train_skip + val_skip
    print(f"\nTotal: {total_seqs} sequences ({train_orig + val_orig} originals + {train_trans + val_trans} transpositions, {total_skip} skipped)")
    print(f"Vocab size: {VOCAB_SIZE}")

    # Shuffle sequences before flattening — prevents transpositions of the same
    # piece from being adjacent, which would cause near-duplicate consecutive batches.
    rng.shuffle(train_seqs)
    rng.shuffle(val_seqs)

    # Flatten
    train_flat = [t for s in train_seqs for t in s]
    val_flat   = [t for s in val_seqs   for t in s]

    print(f"Train: {len(train_seqs)} sequences, {len(train_flat):,} tokens")
    print(f"Val:   {len(val_seqs)} sequences, {len(val_flat):,} tokens")
    print(f"Augmentation factor: {len(train_flat) / max(1, sum(len(s) for s in train_seqs[:train_orig])) :.1f}×" if train_orig > 0 else "")

    torch.save(torch.tensor(train_flat, dtype=torch.long),
               os.path.join(CACHE_DIR, "train.pt"))
    torch.save(torch.tensor(val_flat, dtype=torch.long),
               os.path.join(CACHE_DIR, "val.pt"))

    meta = dict(
        vocab_size=VOCAB_SIZE, composers=COMPOSERS,
        n_train_originals=train_orig, n_val_originals=val_orig,
        n_train_transpositions=train_trans, n_val_transpositions=val_trans,
        n_train=len(train_seqs), n_val=len(val_seqs),
        train_tokens=len(train_flat), val_tokens=len(val_flat),
        transpose_range=list(TRANSPOSE_RANGE),
    )
    with open(os.path.join(CACHE_DIR, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Cached to {CACHE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        mp = os.path.join(CACHE_DIR, "metadata.json")
        if os.path.exists(mp):
            print(json.dumps(json.load(open(mp)), indent=2))
        else:
            print("No cache found. Run prepare.py first.")
    else:
        prepare_data()
        print("\nDone! Ready to train.")
