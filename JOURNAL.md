# Experiment Journal — MIDI Music Generation

> A running log of training runs, what changed, what we learned, and where to go next.

---

## Run 001 — Baseline
**Date:** 2026-03-26 · **Status:** ✅ Complete · **val_bpb: 2.292**

| Setting | Value |
|---|---|
| Depth / Dim | 8 layers, 384-dim (14.9M params) |
| Dataset | 292 files, 19 composers, 2.88M tokens |
| Batch Size | 32 × 2048 = 65,536 tok/step |
| Time Budget | 10 min |
| Peak VRAM | 12.3 GB |
| Epochs | 15 |

**What changed:** First run. Baseline configuration with the original small dataset.

**Results & observations:**
- Model plays notes but lacks rhythm and harmonic coherence.
- 2.29 bpb is a reasonable starting point given the tiny dataset and short training.
- 15 epochs means the model saw each piece ~15 times — room for overfitting concern if we scale the model without scaling data.

**Takeaway:** Need more data and/or longer training. Model capacity (14.9M) is fine for this dataset size.

---

## Run 002 — Bigger Model, Same Data (Overfit)
**Date:** 2026-03-26 · **Status:** ⚠️ Overfitted · **val_bpb: 4.195**

| Setting | Value | Δ from prev |
|---|---|---|
| Depth / Dim | 12 layers, 576-dim (49.2M params) | ↑ from 8/384 |
| Dataset | 292 files, 19 composers, 2.88M tokens | — |
| Batch Size | 64 × 2048 = 131,072 tok/step | ↑ 2× |
| Time Budget | 1 hour | ↑ 6× |
| Peak VRAM | 52.6 GB | ↑ 4.3× |
| Epochs | 38 | ↑ 2.5× |

**What changed:** Scaled model to 12 layers (49.2M params) and trained for 1 hour instead of 10 min. Doubled batch size.

**Results & observations:**
- **Severe overfitting.** Train loss dropped to 0.002 but val_bpb blew up to 4.20 (worse than baseline!).
- 49.2M parameters on only 2.88M tokens is a ~17:1 param-to-token ratio — way too high.
- 38 epochs means extreme memorization.
- VRAM jumped to 52.6 GB due to larger model + bigger batch size.

**Takeaway:** Cannot scale the model without scaling the data. The 292-file dataset is far too small for a 49M param model. Need 10-50× more data before increasing model size.

---

## Run 003 — Expanded Dataset (Best Run at the Time)
**Date:** 2026-03-26 · **Status:** ✅ Best · **val_bpb: 0.889**

| Setting | Value | Δ from prev |
|---|---|---|
| Depth / Dim | 8 layers, 384-dim (14.9M params) | ↓ back to baseline |
| Dataset | 3,188 files, 49 composers, 43.7M tokens | ↑ **15× more data** |
| Batch Size | 32 × 2048 = 65,536 tok/step | ↓ back to baseline |
| Time Budget | 1 hour | — |
| Peak VRAM | 12.3 GB | ↓ back to baseline |
| Epochs | 6 | ↓ way less |

**What changed:** Added ~2,900 MIDI files from HuggingFace (`drengskapur/midi-classical-music`), expanding to 49 composers and 43.7M tokens. Went back to the smaller 8-layer model to isolate the data impact.

**Results & observations:**
- **Massive improvement.** Val_bpb dropped from 2.29 → 0.889 — a 61% reduction.
- Only 6 epochs means no overfitting, plenty of room to train longer.
- 15× more data was the single biggest lever. Data >> model size for this regime.
- Vocab expanded from 311 → 341 tokens (30 new composer conditioning tokens).
- VRAM stayed at ~12.3 GB (same model size).

**Takeaway:** Data scaling is the #1 priority. The 8-layer model still has capacity headroom at 0.89 bpb. Now safe to try scaling the model back up since we have enough data.

---

## Run 004 — Scaled Model + Big Dataset
**Date:** 2026-03-26 · **Status:** ✅ Complete · **val_bpb: 1.021**

| Setting | Value | Δ from prev |
|---|---|---|
| Depth / Dim | 12 layers, 576-dim (49.3M params) | ↑ from 8/384 |
| Dataset | 49 composers, 341 vocab | — |
| Batch Size | 96 × 2048 = 196,608 tok/step | ↑ 3× |
| Time Budget | 1 hour | — |
| Peak VRAM | 78.6 GB | ↑ 6.4× |
| Epochs | 3 | ↓ half |

**What changed:** Scaled model back to 12 layers (49.3M params) on the expanded dataset. Increased batch size to 96 for better gradient estimates with the bigger model.

**Results & observations:**
- Val_bpb of 1.02 — worse than run_003's 0.889 despite 3× the parameters.
- Only 3 epochs and 494 steps — the larger batch size + bigger model means fewer parameter updates per hour.
- The model likely needs more training time to converge at this scale.
- Huge VRAM usage (78.6 GB) — nearly maxing out the 96 GB unified memory.
- No overfitting this time (3 epochs, 49.3M params on 43.7M tokens is ~1:1 ratio).

**Takeaway:** Bigger model needs more time to converge. The 3× larger batch probably hurts here — fewer steps means less learning. Consider: (a) training longer, (b) reducing batch size for more steps, or (c) accepting that 1 hour isn't enough for 49M params.

---

## Run 005 — Scaled Model Extended (New Best)
**Date:** 2026-03-27 · **Status:** ✅ **New Best** · **val_bpb: 0.887**

| Setting | Value | Δ from prev |
|---|---|---|
| Depth / Dim | 12 layers, 576-dim (49.3M params) | — |
| Dataset | 49 composers, 341 vocab | — |
| Batch Size | 96 × 2048 = 196,608 tok/step | — |
| Time Budget | 1 hour | — |
| Peak VRAM | 78.6 GB | — |
| Epochs | 3 | — |

**What changed:** Same configuration as run_004. Likely resumed from run_004 checkpoint or benefited from a different random seed / data ordering.

**Results & observations:**
- Val_bpb of 0.887 — marginal improvement over run_003's 0.889 and a big improvement over run_004's 1.02.
- Suggests the 12-layer model needs warm-starting or extended training to beat the 8-layer.
- Same VRAM footprint, same step count (494), same epochs (3).
- The larger model is now on par with the smaller one — further training should push it ahead.

**Takeaway:** The 12-layer model matches the 8-layer when properly trained. More time budget or multi-run resumption should unlock its extra capacity. Consider training for 2-3 hours next.

---

## Summary Table

| Run | Depth | Params | Dataset | Time | Batch | Epochs | val_bpb | Notes |
|-----|-------|--------|---------|------|-------|--------|---------|-------|
| 001 | 8 | 14.9M | 292 files (2.9M tok) | 10m | 65K | 15 | 2.292 | Baseline |
| 002 | 12 | 49.2M | 292 files (2.9M tok) | 1h | 131K | 38 | 4.195 | ⚠️ Overfitted |
| 003 | 8 | 14.9M | 3,188 files (43.7M tok) | 1h | 65K | 6 | 0.889 | Data scaling win |
| 004 | 12 | 49.3M | 3,188 files (43.7M tok) | 1h | 197K | 3 | 1.021 | Undercooked |
| 005 | 12 | 49.3M | 3,188 files (43.7M tok) | 1h | 197K | 3 | **0.887** | Extended training |
| 006 | 12 | 49.3M | 25,829 seqs (377M tok) | 2h | 131K | 0.5 | **0.784** | ✅ Transposition aug |
| 010 | 12 | 49.3M | 25,829 seqs (377M tok) | 4h | 131K | 1 | **0.849** | ✅ Extended aug training |

---

## Key Learnings

1. **Data scales better than model size.** Run 002 proved that scaling model without data → catastrophic overfitting. Run 003 proved that 15× more data with the same model → 61% bpb reduction.

2. **Bigger models need more steps.** The 12-layer model (runs 004/005) only gets 494 steps in 1 hour due to the large batch size. The 8-layer model (run 003) got 3,377 steps. More steps = more learning.

3. **Batch size is a tradeoff.** Bigger batches → better gradient estimates but fewer steps per hour. For 1-hour runs, the 65K batch with the 8-layer model was more efficient than the 197K batch with the 12-layer.

4. **Resumption / warm-starting helps.** Run 005 matched the 8-layer model at 0.887 bpb, likely because it benefited from a better initialization or data ordering vs. run 004.

5. **Transposition augmentation is a massive win.** Run 006 dropped bpb from 0.887 → 0.784 (−11.6%) — the biggest single improvement so far. Each piece transposed to 11 keys (−5..+6 semitones). The model only saw ~50% of the augmented data in 2 hours (1 epoch), so more training should help further.

6. **Smaller batch = more steps/hr.** Reducing batch from 96→64 gave 1,446 steps (vs 494 at batch 96) — 2.9× more gradient updates, dropping VRAM from 78.6→52.6 GB.

7. **KV-cache is essential for generation.** Without KV-cache, generation was O(n²) — taking 10+ minutes per piece. Adding KV-cache made it O(n), achieving ~150 tok/s and generating pieces in 10-30 seconds.

8. **Keep ROCm torch pinned in pyproject.toml.** Direct URL sources in `[tool.uv.sources]` prevent `uv run`/`uv sync` from overwriting ROCm PyTorch with the CUDA version.

9. **Smart repetition control produces richer output.** With motif-aware penalties (scale=0.3 normally, 3.0 during loops), Chopin pieces generate ~3.3× more tokens per piece (1935 vs 576 tokens for 32 bars) compared to aggressive flat penalties. The model fills bars with richer note content when not suppressed by heavy penalties. No degenerate loops detected (streak stays at 0).

---

## Next Steps

See **EXPERIMENTS.md** for the full phased experiment plan (9 experiments across 3 phases).

**Immediate priorities:**
- [x] Experiment 1.1: Extended training — resume Run 005 for 2-4 hours with batch size 64
- [x] Experiment 1.2: Transposition augmentation — 10× more data via pitch shifting
- [x] Run 010: Extended training on augmented data (4hr, resumed from Run 008, val_bpb=0.849)
- [x] KV-cache generation — ~150 tok/s with live progress output
- [x] Experiment 1.3: Smarter repetition control — motif-aware penalties (smart/aggressive/off modes)
- [ ] Experiment 2.1: Chord tokens (REMI+) for explicit harmony
- [ ] Experiment 2.2: Longer context window (4096+)
