# Experiment Plans — MIDI Music Generation Quality

> Phased experiment roadmap to improve motif coherence, harmonic quality, sectional structure, and emotional direction.
> **Approved:** 2026-03-27 · **Current best:** Run 006, val_bpb=0.784

---

## Overview

| Phase | # | Experiment | Files Changed | Effort | Expected Impact |
|-------|---|-----------|---------------|--------|----------------|
| 1 | 1.1 | Extended Training | train.py (config only) | 🟢 Low | Medium |
| 1 | 1.2 | Transposition Augmentation | prepare.py | 🟢 Low | High |
| 1 | 1.3 | Smarter Repetition Control | generate.py | 🟢 Low | Medium-High |
| 2 | 2.1 | Chord Tokens (REMI+) | prepare.py, generate.py | 🟡 Medium | **Very High** |
| 2 | 2.2 | Longer Context Window | prepare.py, train.py | 🟡 Medium | High |
| 2 | 2.3 | Section Boundary Tokens | prepare.py, generate.py | 🟡 Medium | Medium |
| 3 | 3.1 | Rule-Based Reward Scoring | new: score.py | 🟡 Medium | High |
| 3 | 3.2 | DPO Fine-Tuning | new: dpo_train.py | 🔴 High | Very High |
| 3 | 3.3 | Harmony-Constrained Decoding | generate.py | 🟢 Low | Medium |

**Critical path:** 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → evaluate → decide on Phase 3

---

## Phase 1: Quick Wins

### Experiment 1.1 — Extended Training

**Status:** ✅ Done (Run 006: batch 64, TIME_BUDGET 7200s)
**Hypothesis:** At 3 epochs the model hasn't saturated. More training deepens harmonic/structural learning.
**Confidence:** High

#### Why this should work
- Run 005 trained for only 3 epochs (494 steps) — the model has barely memorized the dataset
- Reducing batch size gives 2× more gradient updates per hour, which matters more than batch quality at this scale
- Run 003 (8-layer) got 3,377 steps in 1 hour vs. Run 005's 494 — more steps = more learning

#### Implementation plan

1. **Change hyperparameters in train.py:**
   ```python
   # train.py — only these constants change
   DEVICE_BATCH_SIZE = 64          # was 96 → fewer tokens/step but 2× more steps
   TOTAL_BATCH_SIZE = 64 * 2048    # = 131,072 tokens per optimizer step
   ```

2. **Change time budget in prepare.py:**
   ```python
   # prepare.py
   TIME_BUDGET = 7200              # was 3600 → 2 hours
   ```

3. **Run with resume:**
   ```bash
   RESUME_RUN=5 uv run python train.py
   ```

4. **If val_bpb improves, try 4 hours:**
   ```python
   TIME_BUDGET = 14400             # 4 hours
   ```

#### What to measure
- val_bpb (target: < 0.85)
- Generate 5 pieces with default settings, listen for:
  - Fewer obviously wrong notes
  - Any emergent motif repetition
  - Harmonic direction (does it sound like it's "going somewhere"?)

#### Expected VRAM / throughput
- 64 × 2048 = 131K tokens/step → ~52 GB VRAM (down from 78.6 GB)
- ~2× more steps per hour → ~988 steps in 2 hours

---

### Experiment 1.2 — Transposition Augmentation

**Status:** ✅ Done (Run 006: val_bpb 0.887 → 0.784, −11.6%)
**Hypothesis:** Transposing each piece to all 12 keys teaches key-invariant harmony — the model learns interval relationships rather than absolute pitch memorization.
**Confidence:** High

#### Why this should work
- Currently each piece exists in exactly one key — the model must learn "C-E-G is consonant" AND "D-F#-A is consonant" as separate facts
- With transposition, the model sees the same harmonic pattern in every key and learns the STRUCTURE (major triad = root + 4 semitones + 3 semitones)
- ~10× more training data from the same source material
- Used by MuseNet (OpenAI) and standard in music ML research

#### Implementation plan

1. **Add transposition to prepare.py (new function):**
   ```python
   PIANO_MIN, PIANO_MAX = 21, 108  # A0 to C8

   def transpose_notes(notes, semitones):
       """Transpose all notes by N semitones. Returns None if any note goes out of range."""
       transposed = []
       for start, pitch, dur, vel in notes:
           new_pitch = pitch + semitones
           if new_pitch < PIANO_MIN or new_pitch > PIANO_MAX:
               return None  # skip this transposition
           transposed.append((start, new_pitch, dur, vel))
       return transposed
   ```

2. **Modify `prepare_data()` to generate transpositions:**
   ```python
   def prepare_data():
       # ... existing file discovery ...
       
       sequences, total_tok, skipped = [], 0, 0
       for fp, comp in entries:
           try:
               # Original
               toks = tokenize_file(fp, comp_map[comp])
               if len(toks) > 10:
                   sequences.append(toks)
                   total_tok += len(toks)
               
               # Transpositions: -5 to +6 semitones (skip 0 = original)
               notes, tpb, bpm, ts_num, ts_den = _parse_midi(fp)
               for shift in range(-5, 7):
                   if shift == 0:
                       continue
                   t_notes = transpose_notes(notes, shift)
                   if t_notes is not None:
                       toks_t = _tokenize_from_notes(t_notes, tpb, bpm, ts_num, ts_den, comp_map[comp])
                       if len(toks_t) > 10:
                           sequences.append(toks_t)
                           total_tok += len(toks_t)
           except Exception as e:
               skipped += 1
       # ... rest unchanged ...
   ```

3. **Factor out `_tokenize_from_notes()` from `tokenize_file()`:**
   - Extract the quantization + token-building logic from `tokenize_file()` into a reusable function that takes pre-parsed notes
   - `tokenize_file()` becomes: parse MIDI → `_tokenize_from_notes()`

4. **Re-tokenize and retrain:**
   ```bash
   uv run python prepare.py          # re-tokenize with transpositions
   uv run python prepare.py --stats  # verify ~10× more tokens
   uv run python train.py            # train fresh (new vocab = same, more data)
   ```

#### What to measure
- Dataset size: should go from ~38M to ~300M+ tokens
- val_bpb (should improve due to more data + better generalization)
- Listen test: chords should sound more "in key", fewer random dissonances
- Test: generate in unusual keys — should sound as good as common keys

#### Risks
- Some transpositions will be filtered out (notes outside piano range) — that's fine
- The validation set should NOT include transpositions of training pieces (same split, then transpose within each split)
- ~10× more data means ~10× fewer epochs per hour — may need longer training

---

### Experiment 1.3 — Smarter Repetition Control

**Status:** ⬜ Not started
**Hypothesis:** The current anti-repetition penalties suppress musical motif repetition along with degenerate loops. A smarter system can distinguish "good repetition" (motifs) from "bad repetition" (stuck loops).
**Confidence:** Medium

#### Evidence
- With penalties off (`--repetition-penalty 1.0 --ngram-penalty 0`), Chopin degenerates to 2-note loops
- With penalties on, output is an "endless stream of new ideas" — no motifs return
- The n-gram penalty at sizes (3,4,5,6,8) penalizes ANY repeated phrase, including musical motifs (a 4-note motif = 16 tokens)

#### Implementation plan

1. **Replace the penalty system in `generate.py` with a hierarchical approach:**

   ```python
   class MotifAwareRepetitionControl:
       """Allow musical repetition, prevent degenerate loops."""
       
       def __init__(self):
           self.bar_history = []        # list of bar fingerprints
           self.motif_library = {}      # fingerprint -> count
           self.consecutive_repeats = 0
       
       def on_bar_complete(self, bar_tokens):
           """Called when a BAR token is generated."""
           fp = self._fingerprint(bar_tokens)
           
           # Track consecutive identical bars (the degenerate case)
           if self.bar_history and self.bar_history[-1] == fp:
               self.consecutive_repeats += 1
           else:
               self.consecutive_repeats = 0
           
           self.bar_history.append(fp)
           self.motif_library[fp] = self.motif_library.get(fp, 0) + 1
       
       def get_penalty_scale(self):
           """Returns penalty multiplier. >1 = increase penalties, <1 = reduce."""
           if self.consecutive_repeats >= 3:
               return 3.0   # strongly penalize — stuck in a loop
           elif self.consecutive_repeats >= 2:
               return 2.0   # getting suspicious
           else:
               return 0.3   # allow natural repetition
       
       def should_boost_motif_return(self, bars_since_last_occurrence):
           """Boost probability of returning to a motif heard 2-8 bars ago."""
           return 2 <= bars_since_last_occurrence <= 8
       
       def _fingerprint(self, tokens):
           """Hash a bar's pitch content (ignoring velocity/timing for similarity)."""
           pitches = tuple(t for t in tokens if is_pitch(t))
           return hash(pitches)
   ```

2. **Modify the generation loop:**
   - Replace the flat `repetition_penalty`, `presence_penalty`, `ngram_penalty` with the `MotifAwareRepetitionControl`
   - When `consecutive_repeats >= 3`: apply strong penalties (current behavior)
   - When `consecutive_repeats < 2`: dramatically reduce penalties to allow musical repetition
   - Add a "motif return bonus": if the model starts generating tokens similar to a phrase from 2-8 bars ago, BOOST those logits slightly

3. **Add CLI flags:**
   ```
   --repetition-mode smart     # new default: hierarchical
   --repetition-mode aggressive # old behavior
   --repetition-mode off        # no penalties
   ```

#### What to measure
- Listen test: do motifs recur? Do pieces have a sense of "theme"?
- Quantitative: count bar-level fingerprints that appear ≥2 times in a piece
- Degeneration test: does it still avoid stuck loops?

---

## Phase 2: Structural Improvements

### Experiment 2.1 — Chord Tokens (REMI+)

**Status:** ⬜ Not started
**Hypothesis:** Explicit chord tokens at bar boundaries give the model a harmonic roadmap. Instead of inferring harmony from raw pitches, it sees `CHORD Cmaj` directly.
**Confidence:** High — this is the **single highest-impact experiment** in the plan.

#### Why this should work
- The model currently has to learn "C4, E4, G4 at the same position = C major chord" purely from co-occurrence statistics
- With chord tokens, it learns chord PROGRESSIONS as a first-class concept: I → IV → V → I
- During generation, the model generates a chord token first, then fills in notes consistent with that chord
- REMI+ (Hsiao et al., 2021) and HAT (Harmony-Aware Transformer) demonstrated significant quality gains

#### Implementation plan

1. **Add chord detection to prepare.py:**

   ```python
   # Chord vocabulary: 12 roots × 5 qualities = 60 tokens + 1 "no chord"
   CHORD_QUALITIES = ['maj', 'min', 'dim', 'aug', 'dom7']
   PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
   NUM_CHORD_TOKENS = 12 * len(CHORD_QUALITIES) + 1  # 61
   CHORD_OFF = COMP_OFF + len(COMPOSERS)  # after composer tokens
   # New VOCAB_SIZE = CHORD_OFF + NUM_CHORD_TOKENS
   
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
   ```

2. **Modify tokenization to insert chord tokens per bar:**

   ```python
   def tokenize_file(filepath, composer_idx):
       # ... existing parsing and quantization ...
       
       # Group notes by bar
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
   ```

3. **Update generate.py:**
   - After generating BAR token, let the model generate a CHORD token
   - The chord token guides subsequent pitch generation naturally (the model learned this association)
   - For harmony-constrained mode (Exp 3.3), can enforce that generated pitches match the generated chord

4. **Update token vocabulary constants:**
   ```python
   CHORD_OFF = COMP_OFF + len(COMPOSERS)        # 341
   NUM_CHORD_TOKENS = 12 * 5 + 1                # 61 (60 chords + no_chord)
   VOCAB_SIZE = CHORD_OFF + NUM_CHORD_TOKENS     # 402
   ```

5. **Retokenize and retrain:**
   ```bash
   rm -rf .midi_cache/
   uv run python prepare.py
   uv run python train.py     # fresh training with new vocab
   ```

#### What to measure
- val_bpb (may go up slightly due to new token types, then down with training)
- Listen test: harmonic coherence should improve dramatically
- Analyze generated chord progressions: do they follow common patterns?
- Compare "in-key percentage" before vs. after

#### Dependencies
- Can be combined with Experiment 1.2 (transposition) — transpose THEN detect chords
- Should install `music21` for robust chord detection, or use the template-matching heuristic above

---

### Experiment 2.2 — Longer Context Window

**Status:** ⬜ Not started
**Hypothesis:** MAX_SEQ_LEN=2048 covers only ~12% of a typical piece. The model can't learn section-level structure because it never sees section boundaries within its context window.
**Confidence:** High

#### Why this should work
- Average piece = 16,829 tokens. At 2048, the model trains on random 2048-token windows — it sees the middle of pieces but rarely sees intro→theme or theme→recap transitions
- Music Transformer (Huang et al., 2018) showed longer context directly improves structural coherence
- RoPE already supports arbitrary lengths (rotary_seq_len = sequence_len × 10)

#### Implementation plan

1. **Try MAX_SEQ_LEN = 4096 first:**
   ```python
   # prepare.py
   MAX_SEQ_LEN = 4096
   
   # train.py — adjust batch to fit VRAM
   DEVICE_BATCH_SIZE = 48    # was 96; 48 × 4096 = 196,608 tokens (same total)
   # or even 32 × 4096 = 131,072 if VRAM tight
   ```

2. **VRAM estimation:**
   | Config | Tokens/step | Est. VRAM | Steps/hr (est.) |
   |--------|------------|-----------|----------------|
   | 96×2048 (current) | 196,608 | 78.6 GB | 494 |
   | 48×4096 | 196,608 | ~80 GB | ~250 |
   | 32×4096 | 131,072 | ~55 GB | ~330 |
   | 24×8192 | 196,608 | ~85 GB | ~125 |

3. **If 4096 shows gains, try 8192:**
   ```python
   MAX_SEQ_LEN = 8192
   DEVICE_BATCH_SIZE = 24    # 24 × 8192 = 196,608
   ```

4. **Retokenize and retrain:**
   ```bash
   rm -rf .midi_cache/
   uv run python prepare.py
   uv run python train.py
   ```

#### What to measure
- val_bpb (should improve with longer context)
- Listen test: does the piece have clearer sections? Intro/body/outro?
- Quantitative: analyze self-similarity of generated pieces (do motifs return after 8-16 bars?)

#### Trade-offs
- Quadratic attention cost (O(n²)) — 4096 is 4× more attention compute than 2048
- Fewer steps per hour — need longer training to compensate
- Can combine with transposition augmentation (more data helps compensate for fewer steps)

---

### Experiment 2.3 — Section Boundary Tokens

**Status:** ⬜ Not started
**Hypothesis:** Explicit section markers teach the model compositional form (ABA, ABAB, etc.).
**Confidence:** Medium — depends heavily on section detection quality.

#### Implementation plan

1. **Implement section detection via self-similarity:**

   ```python
   import numpy as np
   
   def detect_sections(bars_data, min_section_length=4):
       """Detect section boundaries using pitch histogram similarity.
       
       Args:
           bars_data: list of lists, each inner list contains (pos, pitch, dur, vel) for one bar
       Returns:
           list of (bar_start, bar_end, section_label) tuples
       """
       n_bars = len(bars_data)
       if n_bars < 8:
           return [(0, n_bars - 1, 'A')]
       
       # Compute pitch-class histogram per bar
       histograms = []
       for bar in bars_data:
           h = np.zeros(12)
           for _, pitch, dur, _ in bar:
               h[pitch % 12] += dur  # weight by duration
           norm = h.sum()
           if norm > 0:
               h /= norm
           histograms.append(h)
       
       # Self-similarity matrix (cosine similarity)
       H = np.array(histograms)
       norms = np.linalg.norm(H, axis=1, keepdims=True)
       norms[norms == 0] = 1
       H_norm = H / norms
       sim = H_norm @ H_norm.T
       
       # Detect boundaries: bars where similarity to previous bar drops significantly
       boundaries = [0]
       for i in range(1, n_bars):
           if sim[i, i-1] < 0.7:  # threshold for section change
               if i - boundaries[-1] >= min_section_length:
                   boundaries.append(i)
       boundaries.append(n_bars)
       
       # Label sections by similarity clustering
       # If section i is similar to section j (earlier), reuse its label
       sections = []
       label_map = {}
       next_label = 0
       for i in range(len(boundaries) - 1):
           start, end = boundaries[i], boundaries[i + 1]
           section_hist = np.mean(histograms[start:end], axis=0)
           
           # Check similarity to previous sections
           matched_label = None
           for prev_label, prev_hist in label_map.items():
               cos_sim = np.dot(section_hist, prev_hist) / (np.linalg.norm(section_hist) * np.linalg.norm(prev_hist) + 1e-8)
               if cos_sim > 0.85:
                   matched_label = prev_label
                   break
           
           if matched_label is not None:
               label = matched_label
           else:
               label = chr(ord('A') + next_label)
               next_label = min(next_label + 1, 7)  # max 8 sections (A-H)
               label_map[label] = section_hist
           
           sections.append((start, end - 1, label))
       
       return sections
   ```

2. **Add section tokens to vocabulary:**
   ```python
   SECTION_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
   SECTION_OFF = CHORD_OFF + NUM_CHORD_TOKENS  # after chord tokens (if using 2.1)
   # or SECTION_OFF = COMP_OFF + len(COMPOSERS) if NOT using chord tokens
   NUM_SECTION_TOKENS = len(SECTION_LABELS)    # 8
   PHRASE_END_TOKEN = SECTION_OFF + NUM_SECTION_TOKENS  # 1 token
   ```

3. **Modify tokenization to insert section markers:**
   ```python
   # In tokenize_file():
   sections = detect_sections(bars_data)
   current_section_idx = 0
   
   for bar_num in sorted(bars.keys()):
       # Check if we've entered a new section
       while current_section_idx < len(sections):
           start, end, label = sections[current_section_idx]
           if bar_num == start:
               tokens.append(tok_section(label))
               break
           elif bar_num > end:
               current_section_idx += 1
           else:
               break
       
       tokens.append(BAR)
       # ... chord + notes ...
   ```

4. **At generation time, the model naturally generates SECTION_A, SECTION_B, etc.**

#### What to measure
- Listen test: do pieces have clearer structural contrast between sections?
- Quantitative: do generated pieces produce section tokens in meaningful patterns (e.g., ABA, ABAB)?
- Compare with and without section tokens on the same model

#### Risks
- Section detection is imperfect — noisy labels could hurt more than help
- Might want to start with a simpler approach: just PHRASE_END tokens every 4-8 bars

---

## Phase 3: Advanced / Experimental

### Experiment 3.1 — Rule-Based Reward Scoring

**Status:** ⬜ Not started
**Hypothesis:** Computable music theory rules can score generated pieces, enabling best-of-N selection and later RL fine-tuning.
**Confidence:** Medium-High

#### Implementation plan

1. **Create `score.py` with scoring functions:**

   ```python
   def score_midi(tokens) -> dict:
       """Score a generated token sequence on multiple musical quality dimensions."""
       scores = {}
       
       # 1. Harmonic consonance (0-1): % of notes fitting detected key
       scores['harmony'] = _score_harmony(tokens)
       
       # 2. Motif recurrence (0-1): normalized count of repeated phrases
       scores['motif'] = _score_motif_recurrence(tokens)
       
       # 3. Dynamic range (0-1): velocity variance
       scores['dynamics'] = _score_dynamics(tokens)
       
       # 4. Structural balance (0-1): entropy of bar-level pitch histograms
       scores['structure'] = _score_structure(tokens)
       
       # 5. Dissonance penalty (0-1): penalize simultaneous minor 2nds, tritones
       scores['consonance'] = _score_consonance(tokens)
       
       # 6. Pitch range usage (0-1): using a reasonable range, not stuck on 3 notes
       scores['range'] = _score_pitch_range(tokens)
       
       # Weighted aggregate
       weights = {'harmony': 0.25, 'motif': 0.20, 'dynamics': 0.10,
                  'structure': 0.15, 'consonance': 0.20, 'range': 0.10}
       scores['total'] = sum(scores[k] * weights[k] for k in weights)
       
       return scores
   ```

2. **Key scoring functions:**

   ```python
   def _score_harmony(tokens):
       """Use Krumhansl-Schmuckler key-finding algorithm."""
       # Extract all pitches
       pitches = [dec_pitch(t) for t in tokens if is_pitch(t)]
       if len(pitches) < 8:
           return 0.0
       # Pitch class profile
       profile = [0] * 12
       for p in pitches:
           profile[p % 12] += 1
       # Correlate with major/minor key profiles for all 24 keys
       major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
       minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
       best_corr = 0
       for root in range(12):
           rotated = profile[-root:] + profile[:-root]
           for key_profile in [major_profile, minor_profile]:
               corr = _pearson_correlation(rotated, key_profile)
               best_corr = max(best_corr, corr)
       return max(0, best_corr)  # 0-1 range
   
   def _score_motif_recurrence(tokens):
       """Count repeated 4-note phrases (16-token n-grams)."""
       # Extract note groups (POS PITCH DUR VEL = 4 tokens per note)
       note_groups = []
       i = 0
       while i < len(tokens) - 3:
           if is_pos(tokens[i]) and is_pitch(tokens[i+1]) and is_dur(tokens[i+2]) and is_vel(tokens[i+3]):
               note_groups.append(tuple(tokens[i:i+4]))
               i += 4
           else:
               i += 1
       
       # Look for repeated 4-note motifs (16 tokens)
       motif_len = 4  # in note groups
       if len(note_groups) < motif_len * 2:
           return 0.0
       motifs = {}
       for i in range(len(note_groups) - motif_len + 1):
           # Use pitch-only fingerprint (ignore velocity/timing for motif matching)
           pitches = tuple(g[1] for g in note_groups[i:i+motif_len])  # just pitch tokens
           motifs[pitches] = motifs.get(pitches, 0) + 1
       
       # Score: fraction of motifs that appear more than once
       repeated = sum(1 for c in motifs.values() if c >= 2)
       return min(1.0, repeated / max(1, len(motifs) * 0.1))
   
   def _score_consonance(tokens):
       """Penalize simultaneous dissonant intervals (minor 2nd, tritone)."""
       # Group notes by bar and position
       bars = _extract_bars_with_positions(tokens)
       total_intervals, dissonant = 0, 0
       for bar in bars:
           for pos, notes_at_pos in bar.items():
               pitches = sorted(set(n[0] % 12 for n in notes_at_pos))
               for i in range(len(pitches)):
                   for j in range(i + 1, len(pitches)):
                       interval = (pitches[j] - pitches[i]) % 12
                       total_intervals += 1
                       if interval in (1, 6):  # minor 2nd, tritone
                           dissonant += 1
       if total_intervals == 0:
           return 1.0
       return 1.0 - (dissonant / total_intervals)
   ```

3. **Add best-of-N generation mode:**
   ```bash
   # Generate 10 pieces, keep the best-scoring one
   uv run python generate.py --composer chopin --best-of 10
   ```

#### What to measure
- Score distribution across generated pieces
- Do high-scoring pieces actually sound better? (validate the reward function)
- Time cost: generating 10 pieces takes ~10× inference time

---

### Experiment 3.2 — DPO Fine-Tuning

**Status:** ⬜ Not started
**Hypothesis:** Direct Preference Optimization can teach the model to prefer high-quality musical patterns over low-quality ones, without explicit rules at generation time.
**Confidence:** Medium — experimental, but proven in NLP (Rafailov et al., 2023) and music (MusicRL, Google 2024).

#### Prerequisites
- Experiment 3.1 (reward scoring) must be working first
- Base model should already be decent (after Phase 1-2 improvements)

#### Implementation plan

1. **Generate preference pairs:**
   ```python
   # generate_preferences.py
   def generate_preference_dataset(model, n_pairs=500):
       """Generate pairs of pieces and score them to create preference data."""
       pairs = []
       for i in range(n_pairs):
           composer = random.choice(COMPOSERS)
           prompt = [BOS, tok_comp(comp_map[composer]), tok_tempo(bpm_to_bin(120))]
           
           # Generate two pieces from same prompt
           tokens_a = generate(model, prompt, temperature=0.95)
           tokens_b = generate(model, prompt, temperature=0.95)
           
           score_a = score_midi(tokens_a)['total']
           score_b = score_midi(tokens_b)['total']
           
           if abs(score_a - score_b) > 0.05:  # only keep pairs with clear preference
               if score_a > score_b:
                   pairs.append((prompt, tokens_a, tokens_b))  # (prompt, preferred, dispreferred)
               else:
                   pairs.append((prompt, tokens_b, tokens_a))
       return pairs
   ```

2. **Implement DPO loss:**
   ```python
   # dpo_train.py
   def dpo_loss(model, ref_model, preferred, dispreferred, beta=0.1):
       """Direct Preference Optimization loss (Rafailov et al., 2023)."""
       # Log probabilities under current model
       logp_pref = get_log_prob(model, preferred)
       logp_dispref = get_log_prob(model, dispreferred)
       
       # Log probabilities under reference model (frozen)
       with torch.no_grad():
           ref_logp_pref = get_log_prob(ref_model, preferred)
           ref_logp_dispref = get_log_prob(ref_model, dispreferred)
       
       # DPO objective
       log_ratio_pref = logp_pref - ref_logp_pref
       log_ratio_dispref = logp_dispref - ref_logp_dispref
       
       loss = -F.logsigmoid(beta * (log_ratio_pref - log_ratio_dispref))
       return loss.mean()
   ```

3. **Training loop:**
   - Freeze a copy of the model as `ref_model`
   - Fine-tune the main model with DPO loss for ~100-500 steps
   - Use low learning rate (1/10 of pretraining LR) to avoid catastrophic forgetting
   - Monitor val_bpb to ensure it doesn't degrade much (some increase is OK)

#### What to measure
- Does val_bpb stay reasonable? (small increase OK, >1.5× is bad)
- Do reward scores improve post-DPO?
- Listen test: does the music sound "more intentional"?

#### Risks
- Mode collapse: model might only generate one "safe" pattern
- Reward hacking: model might find ways to score high that sound bad
- Mitigation: KL regularization (built into DPO via reference model)

---

### Experiment 3.3 — Harmony-Constrained Decoding

**Status:** ⬜ Not started
**Hypothesis:** At generation time, detecting the current key and down-weighting out-of-key pitches eliminates random dissonance.
**Confidence:** Medium — addresses symptoms, not root cause.

#### Implementation plan

1. **Add key detection to generate.py:**
   ```python
   def detect_key_from_recent(tokens, window=128):
       """Detect the current musical key from recent tokens using Krumhansl profiles."""
       recent = tokens[-window:] if len(tokens) > window else tokens
       pitches = [dec_pitch(t) for t in recent if is_pitch(t)]
       if len(pitches) < 4:
           return None, None  # not enough data
       
       profile = [0] * 12
       for p in pitches:
           profile[p % 12] += 1
       
       major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
       minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
       
       best_corr, best_root, best_mode = -1, 0, 'major'
       for root in range(12):
           rotated = profile[-root:] + profile[:-root]
           for mode, key_prof in [('major', major), ('minor', minor)]:
               corr = _pearson(rotated, key_prof)
               if corr > best_corr:
                   best_corr = corr
                   best_root = root
                   best_mode = mode
       
       return best_root, best_mode
   ```

2. **Apply soft pitch mask during generation:**
   ```python
   # In the generate() loop, after computing logits:
   if harmony_constrain:
       root, mode = detect_key_from_recent(generated)
       if root is not None:
           # Determine in-key pitches
           if mode == 'major':
               scale = [0, 2, 4, 5, 7, 9, 11]  # major scale intervals
           else:
               scale = [0, 2, 3, 5, 7, 8, 10]  # natural minor
           
           in_key_pcs = set((root + s) % 12 for s in scale)
           
           # Penalize out-of-key pitch tokens
           for midi_pitch in range(128):
               tok_id = tok_pitch(midi_pitch)
               if tok_id < logits.size(-1):
                   if midi_pitch % 12 not in in_key_pcs:
                       logits[0, tok_id] -= 2.0  # soft penalty, not hard mask
   ```

3. **Add CLI flag:**
   ```
   --harmony-constrain        # enable key-aware pitch masking
   --harmony-strength 2.0     # penalty for out-of-key notes (default: 2.0)
   --harmony-window 128       # tokens to analyze for key detection
   ```

#### What to measure
- Listen test: do chords sound more consonant?
- Does it reduce musical variety too much? (risk of sounding "bland")
- Compare harmony scores (from 3.1) with and without constraint

#### Trade-offs
- Quick to implement, immediate results
- But doesn't teach the model anything — the model is still "wrong", we're just masking it
- Best as a bridge measure while training-side improvements (2.1, 2.2) take effect

---

## Experiment Tracking Template

When running experiments, log results in JOURNAL.md using this format:

```markdown
## Run NNN — Experiment X.Y: [Name]
**Date:** YYYY-MM-DD · **Status:** ⬜/✅/⚠️ · **val_bpb: X.XXX**

| Setting | Value | Δ from prev |
|---|---|---|
| Key change | value | what changed |

**What changed:** [describe the experiment]

**Results & observations:**
- val_bpb: X.XXX (was Y.YYY)
- Listen test: [subjective notes]
- Quantitative: [metrics from score.py if available]

**Takeaway:** [what we learned, what to try next]
```
