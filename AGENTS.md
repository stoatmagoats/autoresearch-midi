# AGENTS.md — LLM Guide for MIDI Music Generation

## Overview

**autoresearch-midi** is a GPT-based MIDI music generation system. A transformer model is trained on classical piano MIDI files using REMI-style tokenization, then generates new composer-conditioned pieces as standard MIDI files. Built on the [autoresearch](https://github.com/andyluo7/autoresearch) framework, adapted for AMD ROCm (Radeon 8060S / Strix Halo).

## Project Structure

```
prepare.py          — MIDI tokenizer (REMI-style), dataloader, evaluation
train.py            — GPT model, MuonAdamW optimizer, training loop
generate.py         — Autoregressive sampling with KV-cache → MIDI file output
analyze_midi.py     — Dataset quality analysis (flags bad files for removal)
download_hf_midi.py — Downloads MIDI datasets from HuggingFace
JOURNAL.md          — Experiment journal: per-run tweaks, results, and learnings
EXPERIMENTS.md      — Phased experiment plans for improving music quality
docs/plans/         — Approved design documents for major changes
midi_files/         — Training data: ~2,300 classical piano MIDI files (49 composers)
generated/          — Output directory for generated MIDI files
runs/               — Training run history (checkpoint.pt, config.json, run.log per run)
checkpoint.pt       — Symlink → latest run's checkpoint
.midi_cache/        — Cached tokenized data (train.pt, val.pt, metadata.json)
```

## Architecture

### Tokenization (prepare.py)
REMI-style encoding with **341 tokens**:

| Token Range | Count | Description |
|-------------|-------|-------------|
| 0–3         | 4     | PAD, BOS, EOS, BAR |
| 4–35        | 32    | Position within bar (16th-note quantization) |
| 36–163      | 128   | MIDI pitch (0–127) |
| 164–227     | 64    | Duration (1–64 in 16th notes) |
| 228–259     | 32    | Velocity bins |
| 260–291     | 32    | Tempo bins (40–240 BPM) |
| 292–340     | 49    | Composer conditioning tokens |

Each note = 4 tokens: `POS PITCH DUR VEL`
Sequence format: `BOS COMPOSER TEMPO BAR [notes...] BAR [notes...] ... EOS`

### Model (train.py)
- **GPT** with RoPE, QK-norm, GQA, value residual (ResFormer), ReLU² MLP
- **MuonAdamW** hybrid optimizer (Muon for 2D matrices, AdamW for rest)
- Config: 12 layers, 576-dim, 9 heads (HEAD_DIM=64), ~49.2M parameters
- ROCm: uses PyTorch SDPA (AOTriton), torch.compile enabled (with monkey-patch)

### Dataset
- **2,307 pieces** from 49 classical composers (quality-filtered from ~3,200 originals)
- **Transposition augmentation**: each piece transposed to −5..+6 semitones (~11× data)
- **377M train tokens** / **42M val tokens** (25,829 sequences total)
- Train/val split: 2,077/230 original pieces (90/10), augmented independently
- Source: original 292 files + `drengskapur/midi-classical-music` from HuggingFace
- Filtering: Removed 896 files (single-velocity/synthetic, corrupt, extreme density)
- Available composers: `albeniz, alkan, bach, balakir, barber, bartok, beeth, borodin, brahms, burgm, busoni, chopin, clementi, copland, cpe_bach, czerny, debussy, dvorak, faure, field, franck, gershwin, ginastera, granados, grieg, haendel, haydn, heller, holst, hummel, joplin, kuhlau, liszt, massenet, mendelssohn, mozart, muss, poulenc, prokofiev, rachmaninoff, ravel, satie, schubert, schumann, sibelius, stravinsky, tschai, vivaldi, wagner`

## Running

### One-time setup
```bash
# Install all dependencies (ROCm torch+triton are configured in pyproject.toml)
uv sync

# Tokenize MIDI files (creates .midi_cache/)
uv run python prepare.py
```

### Training (~2 hours)

```bash
# Train the model (auto-creates runs/run_NNN/ with checkpoint, config, log)
uv run python train.py

# Resume training from a previous run
RESUME_RUN=6 uv run python train.py

# Detachable training (survives SSH disconnect):
tmux new -s train 'uv run python train.py'
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train

# Pause training: Ctrl+C (saves checkpoint, prints resume command)

# Check results
grep "^val_bpb:\|^peak_vram_mb:" runs/run_NNN/run.log
```

### Generation
```bash
# Generate a Chopin-style piece
uv run python generate.py --composer chopin

# Generate from a specific run
uv run python generate.py --run 3 --composer bach

# List all training runs
uv run python generate.py --list-runs

# Generate 3 Bach pieces with custom settings
uv run python generate.py --composer bach --n 3 --temperature 0.85 --top-k 40

# Control length: 32 bars (~30 seconds at 120 BPM)
uv run python generate.py --composer chopin --bars 32

# Longer piece: 64 bars with more token budget
uv run python generate.py --bars 64 --max-tokens 8192

# Random composer, save to specific file
uv run python generate.py --output my_piece.mid

# All options
uv run python generate.py --help
```

### Playback with TiMidity++
```bash
# Play directly through speakers
timidity generated/chopin_000.mid

# Convert to WAV
timidity generated/chopin_000.mid -Ow -o output.wav

# Play louder with reverb
timidity -A200 -EFreverb=f generated/chopin_000.mid
```

**TiMidity++ setup**: Requires a soundfont. Add to `/etc/timidity/timidity.cfg`:
```
soundfont /usr/share/soundfonts/FluidR3_GM.sf2
```

## Key Hyperparameters (train.py)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `DEPTH` | 12 | Number of transformer layers |
| `ASPECT_RATIO` | 48 | `model_dim = DEPTH * ASPECT_RATIO` |
| `HEAD_DIM` | 64 | Head dimension (9 heads at 576-dim) |
| `DEVICE_BATCH_SIZE` | 64 | Per-device batch size |
| `TOTAL_BATCH_SIZE` | 131072 | Tokens per optimizer step (64×2048) |
| `TIME_BUDGET` | 7200 | Training time in seconds (2 hours) |
| `MATRIX_LR` | 0.04 | Learning rate for Muon params |
| `WARMDOWN_RATIO` | 0.5 | Fraction of budget for LR cooldown |

## Generation Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--temperature` | 0.95 | Sampling temperature (lower = more conservative) |
| `--top-k` | 50 | Top-k sampling |
| `--top-p` | 0.95 | Nucleus sampling threshold |
| `--max-tokens` | 4096 | Maximum tokens to generate |
| `--bars` | None | Stop after N bars (e.g. 32 ≈ 30s at 120 BPM) |
| `--tempo` | 120 | Playback tempo in BPM |
| `--repetition-mode` | smart | `smart` (motif-aware), `aggressive` (old flat), `off` |
| `--repetition-penalty` | 1.2 | Multiplicative penalty for repeated pitch/dur/vel tokens |
| `--presence-penalty` | 0.3 | Additive penalty for recently seen content tokens |
| `--ngram-penalty` | 2.0 | Penalty for repeating n-gram patterns (0=off) |
| `--motif-bonus` | 0.4 | Logit bonus for pitches from 2-8 bars ago (smart mode) |

## AMD ROCm Specifics
- **GPU**: AMD Radeon 8060S (gfx1151, RDNA 3.5, 96GB unified memory)
- **PyTorch**: `2.9.1+rocm7.2.1` from `repo.radeon.com` (configured in `pyproject.toml` via `[tool.uv.sources]`)
- **Triton**: `3.5.1+rocm7.2.1` (ROCm-specific build, also in `pyproject.toml`)
- **ROCm SDK**: 7.2.0 (system-level, via `pacman -S rocm-hip-sdk`)
- **Attention**: PyTorch SDPA with experimental AOTriton (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`)
- **torch.compile**: Enabled with monkey-patch (fixes ZeroDivisionError in InductorBenchmarker, ~1.2× speedup)
- **Generation**: KV-cache enabled (~150 tok/s), with live progress output
- Peak VRAM: ~35 GB (training, batch 64)

## Metric

**`val_bpb`** (bits per token) — lower is better. Cross-entropy loss on held-out pieces converted to bits. Current best: **0.849** (Run 10, 49.2M params, 4hr training with transposition augmentation).

## Improving Quality

See **EXPERIMENTS.md** for the full phased experiment plan (9 experiments across 3 phases). Summary:

### Phase 1: Quick Wins
1. **Extended training** — resume from Run 005, train 2-4 hours with smaller batch for more steps
2. **Transposition augmentation** — transpose each piece to all 12 keys (~10× more data)
3. **Smarter repetition control** — motif-aware penalties that allow musical repetition while preventing loops

### Phase 2: Structural
4. **Chord tokens (REMI+)** — add explicit harmony tokens at bar boundaries (highest expected impact)
5. **Longer context (4096+)** — see more of each piece during training for better structure
6. **Section boundary tokens** — mark structural sections (A, B, etc.) in training data

### Phase 3: Advanced
7. **Rule-based reward scoring** — music theory scoring for best-of-N selection
8. **DPO fine-tuning** — preference optimization using scored piece pairs
9. **Harmony-constrained decoding** — key-aware pitch masking at generation time
