# MIDI Music Generation — Design Document

**Date:** 2026-03-26
**Status:** Approved

## Goal
Train a small GPT model on 292 classical piano MIDI files to generate composer-conditioned, multi-track MIDI output with tempo/dynamics fidelity.

## Tokenization: REMI-style
Each note → 4 tokens: `POS PITCH DUR VEL`. Pieces structured as:
```
BOS COMPOSER TEMPO BAR [POS PITCH DUR VEL ...] BAR [POS PITCH DUR VEL ...] ... EOS
```

**Vocab (~311 tokens):**
| Range | Count | Description |
|-------|-------|-------------|
| 0–3 | 4 | PAD, BOS, EOS, BAR |
| 4–35 | 32 | Position within bar (16th-note grid) |
| 36–163 | 128 | MIDI pitch (0–127) |
| 164–227 | 64 | Duration (1–64 16th notes) |
| 228–259 | 32 | Velocity bins |
| 260–291 | 32 | Tempo bins (40–240 BPM) |
| 292–310 | 19 | Composer conditioning tokens |

## Architecture
- Reuse GPT from autoresearch (RoPE, QK-norm, value residual, MuonAdamW)
- DEPTH=8, HEAD_DIM=64, ASPECT_RATIO=48 → model_dim=384, 6 heads
- ~15M parameters (appropriate for ~500K–1M token dataset)

## Files Modified/Created
| File | Action |
|------|--------|
| `prepare.py` | Rewrite: MIDI tokenizer, dataloader, eval |
| `train.py` | Adapt: vocab size, model size, batch size, checkpoint saving |
| `generate.py` | New: inference with composer conditioning → MIDI output |

## Evaluation
`val_bpb` = cross-entropy in bits per token on held-out pieces (10% split).
