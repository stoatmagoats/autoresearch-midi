#!/usr/bin/env python3
"""Analyze MIDI dataset quality and identify files to remove."""
import os
import sys

from prepare import MIDI_DIR, tokenize_file, _parse_midi, COMPOSERS
from collections import defaultdict
import statistics

def analyze_file(filepath):
    """Return quality metrics for a single MIDI file."""
    try:
        notes, tpb, bpm, ts_num, ts_den = _parse_midi(filepath)
    except Exception as e:
        return {"error": str(e)}

    if not notes:
        return {"error": "no notes"}

    n_notes = len(notes)
    pitches = [n[1] for n in notes]
    velocities = [n[3] for n in notes]
    durations_ticks = [n[2] for n in notes]
    
    # Duration of piece in seconds (approximate)
    if tpb > 0 and bpm > 0:
        total_ticks = max(n[0] + n[2] for n in notes) - min(n[0] for n in notes)
        duration_sec = total_ticks / tpb / (bpm / 60)
    else:
        duration_sec = 0

    unique_pitches = len(set(pitches))
    pitch_range = max(pitches) - min(pitches)
    avg_velocity = statistics.mean(velocities) if velocities else 0
    
    # Note density (notes per second)
    note_density = n_notes / max(duration_sec, 0.1)
    
    # Check for single-velocity (likely quantized/synthetic)
    vel_unique = len(set(velocities))
    
    return {
        "n_notes": n_notes,
        "unique_pitches": unique_pitches,
        "pitch_range": pitch_range,
        "duration_sec": round(duration_sec, 1),
        "note_density": round(note_density, 1),
        "avg_velocity": round(avg_velocity, 1),
        "vel_unique": vel_unique,
        "bpm": round(bpm, 1),
    }


def main():
    composers = sorted(d for d in os.listdir(MIDI_DIR) if os.path.isdir(os.path.join(MIDI_DIR, d)))
    
    all_stats = []
    bad_files = []
    
    for comp in composers:
        cdir = os.path.join(MIDI_DIR, comp)
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(('.mid', '.midi')):
                continue
            fp = os.path.join(cdir, fn)
            stats = analyze_file(fp)
            stats["composer"] = comp
            stats["file"] = fn
            stats["path"] = fp
            all_stats.append(stats)
            
            # Flag bad files
            reasons = []
            if "error" in stats:
                reasons.append(f"error: {stats['error']}")
            else:
                if stats["n_notes"] < 20:
                    reasons.append(f"too few notes ({stats['n_notes']})")
                if stats["duration_sec"] < 5:
                    reasons.append(f"too short ({stats['duration_sec']}s)")
                if stats["unique_pitches"] < 4:
                    reasons.append(f"too few unique pitches ({stats['unique_pitches']})")
                if stats["pitch_range"] < 6:
                    reasons.append(f"tiny pitch range ({stats['pitch_range']} semitones)")
                if stats["note_density"] > 100:
                    reasons.append(f"extreme density ({stats['note_density']} notes/sec)")
                if stats["vel_unique"] == 1 and stats["n_notes"] > 50:
                    reasons.append(f"single velocity (likely synthetic)")
                if stats["n_notes"] > 50000:
                    reasons.append(f"extremely long ({stats['n_notes']} notes)")
            
            if reasons:
                bad_files.append((fp, reasons, stats))
    
    # Summary
    good = [s for s in all_stats if "error" not in s]
    print(f"=== Dataset Analysis ===")
    print(f"Total files: {len(all_stats)}")
    print(f"Parseable: {len(good)}")
    print(f"Parse errors: {len(all_stats) - len(good)}")
    print()
    
    if good:
        notes_list = [s["n_notes"] for s in good]
        dur_list = [s["duration_sec"] for s in good]
        pitch_list = [s["unique_pitches"] for s in good]
        
        print(f"Notes per file:     min={min(notes_list)}, median={int(statistics.median(notes_list))}, max={max(notes_list)}")
        print(f"Duration (sec):     min={min(dur_list)}, median={statistics.median(dur_list)}, max={max(dur_list)}")
        print(f"Unique pitches:     min={min(pitch_list)}, median={int(statistics.median(pitch_list))}, max={max(pitch_list)}")
    
    print(f"\n=== Flagged for Removal: {len(bad_files)} files ===")
    
    # Group by reason
    reason_counts = defaultdict(int)
    for _, reasons, _ in bad_files:
        for r in reasons:
            category = r.split("(")[0].strip()
            reason_counts[category] += 1
    
    print("\nBreakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print("\nFlagged files:")
    for fp, reasons, stats in sorted(bad_files, key=lambda x: x[0]):
        rel = os.path.relpath(fp, MIDI_DIR)
        print(f"  {rel:60s}  {', '.join(reasons)}")
    
    # Write removal script
    with open("/tmp/remove_bad_midi.sh", "w") as f:
        f.write("#!/bin/bash\n# Remove low-quality MIDI files\n")
        f.write(f"# {len(bad_files)} files flagged for removal\n\n")
        for fp, reasons, _ in sorted(bad_files):
            f.write(f"rm \"{fp}\"  # {', '.join(reasons)}\n")
    os.chmod("/tmp/remove_bad_midi.sh", 0o755)
    print(f"\nRemoval script written to /tmp/remove_bad_midi.sh")
    print(f"Review it, then run: bash /tmp/remove_bad_midi.sh")


if __name__ == "__main__":
    main()
