# Design Doc: Music Quality Improvement Experiments

> **Status:** ✅ APPROVED (2026-03-27)
> **Date:** 2026-03-27
> **Goal:** Improve motif coherence, harmonic quality, sectional structure, and emotional direction in generated MIDI music.

---

## Problem Statement

The model (Run 005, 49.3M params, val_bpb=0.887) generates natural-sounding note sequences but lacks:

1. **Harmonic coherence** — both wrong notes (dissonance) and lack of harmonic direction
2. **Motif repetition/development** — endless stream of new ideas, no recurring themes
3. **Sectional structure** — vague intro/body/outro, nothing well-defined
4. **Emotional arc** — aimless musical character

Root cause diagnosis:
- The model hasn't deeply internalized musical structure (degenerates to note-loops without penalties)
- Only 3 epochs of training — each piece seen just 3 times
- MAX_SEQ_LEN=2048 covers only ~12% of average piece (16,829 tokens) — can't learn full-piece structure
- No explicit harmonic information in tokenization — model must infer key/chord from raw pitches
- Anti-repetition penalties are band-aid that also suppresses musical motif repetition

---

## Phased Experiment Plan

### Phase 1: Quick Wins (1-3 runs each, ~1 day total)

> Goal: Maximize improvement with minimal code changes. Each can be tested independently.

#### Experiment 1.1: Extended Training (High Confidence)

**Hypothesis:** The model at 3 epochs hasn't saturated. More exposure to the data will deepen harmonic and structural learning.

**Changes:**
- Resume from Run 005 checkpoint
- Train for 2 hours (TIME_BUDGET=7200), then try 4 hours
- Reduce batch size to 64×2048=131,072 for more steps per hour (~2× more gradient updates)

**Expected outcome:** Lower val_bpb, richer harmonic patterns, more natural phrasing. This is the lowest-risk, highest-confidence experiment.

**Measures of success:**
- val_bpb < 0.85
- Listen test: fewer obviously wrong notes

---

#### Experiment 1.2: Data Augmentation via Transposition (High Confidence)

**Hypothesis:** The model sees each piece in only one key. Transposing every piece to all 12 keys teaches key-invariant harmonic relationships (e.g., "a 4th above the root sounds consonant" rather than memorizing "C→F is good").

**Changes (prepare.py):**
- For each MIDI file, create 11 transposed copies (±1 to ±6 semitones, skipping 0)
- Filter: skip transpositions where notes go out of piano range (21-108)
- This multiplies effective dataset from ~38M to ~300M+ tokens

**Expected outcome:** Better harmonic generalization, fewer dissonant chords, better performance on underrepresented composers.

**Literature basis:** Used by MuseNet (OpenAI), standard practice in music ML. Well-proven.

**Measures of success:**
- val_bpb improvement (more data = better)
- Listen test: chords sound more "in key"

---

#### Experiment 1.3: Smarter Repetition Control (Medium Confidence)

**Hypothesis:** The current anti-repetition penalties treat all repetition equally. Music needs micro-repetition (motifs = 4-16 note groups) but not degenerate looping (same 2 notes forever). A smarter system should:
- Allow phrase-level repetition (16-64 tokens ≈ 4-16 notes)
- Penalize only exact bar-copy or very short loops (≤8 tokens)
- Use a "cooldown" instead of permanent penalty

**Changes (generate.py):**
- Replace flat n-gram penalty with **hierarchical repetition control**:
  - Allow exact motif repetition if there's been ≥1 bar of different material between occurrences
  - Only penalize *consecutive* identical bars (the degenerate case)
  - Add a "motif memory" that tracks phrase fingerprints and BOOSTS repetition of motifs heard 1-3 bars ago (encouraging thematic return)
- Reduce presence_penalty window from 256 to 64 tokens (only penalize very recent tokens)

**Expected outcome:** Output retains motifs while avoiding degenerate loops.

**Measures of success:**
- Listen test: identifiable recurring themes
- Quantitative: count distinct 8-16 token phrases that appear ≥2 times in output

---

### Phase 2: Structural Improvements (requires prepare.py + retrain, ~1-2 days)

> Goal: Give the model explicit musical information it currently lacks.

#### Experiment 2.1: Chord Tokens in Tokenization (High Confidence)

**Hypothesis:** Adding explicit chord tokens at bar boundaries gives the model a "harmonic roadmap." Instead of inferring "these pitches at this position form a C major chord" from raw data, it sees `CHORD_Cmaj` directly.

**Changes (prepare.py):**
- Use a chord detection algorithm (e.g., from `music21` or simple interval analysis) to identify the dominant chord per bar
- Add CHORD tokens (root × quality = 12 × 4 = 48 new tokens) at the start of each bar: `BAR CHORD_root CHORD_quality [notes...]`
- This is the REMI+ approach from the literature (Hsiao et al., 2021; von Rütte et al., 2023)

**Token format change:**
```
Before: BAR POS PITCH DUR VEL POS PITCH DUR VEL ...
After:  BAR CHORD_ROOT CHORD_QUAL POS PITCH DUR VEL POS PITCH DUR VEL ...
```

**Expected outcome:** Much better harmonic coherence. The model learns chord progressions (I→IV→V→I) explicitly. At generation time, it naturally stays in key.

**Trade-offs:**
- Requires chord detection in prepare.py (need `music21` or custom heuristic)
- Increases vocab from 341 to ~389 tokens
- Requires full re-tokenization and retraining

**Literature basis:** Used in REMI+, HAT (Harmony-Aware Transformer), standard in state-of-the-art music generation.

---

#### Experiment 2.2: Longer Context Window (High Confidence)

**Hypothesis:** At MAX_SEQ_LEN=2048, the model sees ~12% of a piece during training. Doubling to 4096 lets it see ~24%, enough to learn phrase-level structure (intro→theme, A→B transitions). At 8192, it sees ~48%.

**Changes (prepare.py + train.py):**
- Increase MAX_SEQ_LEN to 4096 (or 8192)
- Reduce DEVICE_BATCH_SIZE proportionally to fit in VRAM (e.g., 48×4096 ≈ same memory as 96×2048)
- RoPE already supports this (rotary_seq_len = sequence_len * 10)

**Expected outcome:** Better sense of structure — the model can learn "after 8 bars of theme A, transition to something different" because it can see those 8 bars in context.

**Trade-offs:**
- 4096 context: ~2× slower per step, may need smaller batch
- 8192 context: ~4× slower, significantly reduced batch
- Attention is O(n²) without flash attention (ROCm uses SDPA, which handles this)

**Literature basis:** Music Transformer (Huang et al., 2018) showed that longer context directly improves structural coherence.

---

#### Experiment 2.3: Section/Phrase Boundary Tokens (Medium Confidence)

**Hypothesis:** Adding explicit section markers (`SECTION_A`, `SECTION_B`, `BRIDGE`, `CODA`) and phrase boundaries (`PHRASE_END`) teaches the model compositional form.

**Changes (prepare.py):**
- Detect section boundaries via self-similarity matrix analysis (compute pitch histogram per bar, find transitions)
- Label sections as A, B, C... based on similarity clusters
- Add ~8-10 new structure tokens

**Token format change:**
```
Before: BAR [notes] BAR [notes] BAR [notes] ...
After:  SECTION_A BAR [notes] BAR [notes] PHRASE_END BAR [notes] SECTION_B BAR [notes] ...
```

**Expected outcome:** The model learns "after SECTION_A plays for N bars, generate SECTION_B" and "bring back SECTION_A material for the recap."

**Trade-offs:**
- Section detection is imperfect — noisy labels could confuse the model
- Relatively experimental compared to chord tokens

---

### Phase 3: Advanced / Experimental (multi-day, higher risk/reward)

> Goal: Push beyond standard approaches. Consider only after Phase 1-2 show gains.

#### Experiment 3.1: Rule-Based Reward Scoring (Medium-High Confidence)

**Hypothesis:** Instead of full RLHF (which requires human evaluators), use computable music theory rules as a reward signal for rejection sampling or best-of-N selection.

**Approach:**
- Define a `score_midi(tokens)` function that evaluates:
  - **Harmonic consonance** (% of notes that fit detected key/chord)
  - **Motif recurrence** (count of repeated 4-8 note phrases)
  - **Dynamic range** (velocity variance — penalize flat dynamics)
  - **Structural balance** (entropy of bar-level pitch histograms across the piece)
  - **Dissonance score** (count of minor 2nds, tritones in simultaneous notes)
- Use this for **best-of-N sampling**: generate N pieces, keep the highest-scoring one
- Later: use as reward signal for REINFORCE-style fine-tuning or DPO

**Expected outcome:** Immediate quality improvement via selection. Can later be used to fine-tune the model itself.

**Literature basis:** MusicRL (Google, 2024) showed that even simple reward functions significantly improve music quality when used for RL fine-tuning.

---

#### Experiment 3.2: RLHF / DPO Fine-Tuning (Experimental)

**Hypothesis:** After training a good base model (Phase 1-2), fine-tune with preference optimization to align with musical quality.

**Approach:**
- Generate pairs of pieces from the same prompt (composer + tempo)
- Score them with the rule-based reward from 3.1
- Use DPO (Direct Preference Optimization) to fine-tune — simpler than PPO-based RLHF
- DPO directly optimizes: P(preferred | prompt) > P(dispreferred | prompt)

**Expected outcome:** Model internalizes what makes "good" music vs "bad" music without explicit rules at generation time.

**Trade-offs:**
- Requires generating and scoring hundreds of piece pairs
- DPO can cause mode collapse if not careful (regularization needed)
- Most effective after the base model is already decent

---

#### Experiment 3.3: Harmony-Constrained Decoding (Medium Confidence)

**Hypothesis:** At generation time, detect the current key/chord context and mask or down-weight notes that don't fit.

**Approach:**
- After each bar, analyze the last N notes to detect the current key (Krumhansl-Schmuckler algorithm or simple pitch-class histogram)
- Apply a soft mask to pitch tokens: in-key notes get +0, out-of-key notes get −2.0 penalty
- Allow modulations by resetting key detection every 4-8 bars

**Expected outcome:** Eliminates dissonant notes while preserving the model's learned style.

**Trade-offs:**
- Addresses symptoms (dissonance) rather than root cause
- Could make output sound "too safe" / bland
- Best combined with chord tokens (2.1) so the model learns harmony natively

---

## Recommended Experiment Order

| Priority | Experiment | Effort | Risk | Expected Impact |
|----------|-----------|--------|------|----------------|
| 🥇 1 | 1.1 Extended training | Low | Low | Medium |
| 🥇 2 | 1.2 Transposition augmentation | Low | Low | High |
| 🥇 3 | 1.3 Smarter repetition control | Low | Medium | Medium-High |
| 🥈 4 | 2.1 Chord tokens | Medium | Low | **Very High** |
| 🥈 5 | 2.2 Longer context (4096) | Medium | Low | High |
| 🥈 6 | 2.3 Section boundary tokens | Medium | Medium | Medium |
| 🥉 7 | 3.1 Rule-based reward scoring | Medium | Medium | High |
| 🥉 8 | 3.2 DPO fine-tuning | High | High | Potentially Very High |
| 🥉 9 | 3.3 Harmony-constrained decoding | Low | Medium | Medium |

> **Recommended critical path:** 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → evaluate → decide on Phase 3

---

## Success Criteria

After Phase 1-2, generated pieces should:
- [ ] Have identifiable recurring motifs (same melody fragment appears ≥2 times)
- [ ] Maintain a consistent key/harmony for ≥8 bars at a time
- [ ] Show clear textural contrast between sections
- [ ] Have dynamic variation (quiet vs. loud passages)
- [ ] Sound "intentional" rather than "random" to a casual listener
- [ ] val_bpb ≤ 0.80

---

## Open Questions

1. Should we try compound tokenization (REMI→Compound Word) to shorten sequences? This would let 4096 tokens cover more music.
2. Is the dataset quality actually consistent? Some HuggingFace MIDI files might be low-quality transcriptions.
3. Should we fine-tune per-composer or keep the multi-composer setup? Single-composer models might learn style better.
