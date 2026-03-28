#!/usr/bin/env python3
"""
Download MIDI files from drengskapur/midi-classical-music HuggingFace dataset
and organize them into midi_files/<composer>/ subdirectories.

Only adds files for composers that already exist in our midi_files/ directory,
plus creates new composer directories for new ones found in the dataset.

Usage: python /tmp/download_hf_midi.py
"""
import os
import sys
import shutil
from pathlib import Path

MIDI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "../home/jonathan/Documents/projects/autoresearch/midi_files")
# Fix path to be absolute
MIDI_DIR = "/home/jonathan/Documents/projects/autoresearch/midi_files"
HF_DOWNLOAD_DIR = "/tmp/hf_midi_download"

# Mapping from HF dataset composer names to our directory names
# The HF dataset uses full names like "bach", "beethoven", "chopin"
# Our directories use: albeniz, bach, balakir, beeth, borodin, brahms, burgm, 
#                       chopin, debussy, granados, grieg, haydn, liszt, 
#                       mendelssohn, mozart, muss, schubert, schumann, tschai
COMPOSER_MAP = {
    # Exact matches (HF name -> our dir name)
    "albeniz": "albeniz",
    "bach": "bach",
    "beethoven": "beeth",
    "borodin": "borodin", 
    "brahms": "brahms",
    "burgmuller": "burgm",
    "chopin": "chopin",
    "debussy": "debussy",
    "grieg": "grieg",
    "haydn": "haydn",
    "liszt": "liszt",
    "mendelsonn": "mendelssohn",  # note: HF has typo "mendelsonn"
    "mendelssohn": "mendelssohn",
    "mozart": "mozart",
    "schubert": "schubert",
    "schumann": "schumann",
    # New composers to add
    "alkan": "alkan",
    "bartok": "bartok",
    "barber": "barber",
    "busoni": "busoni",
    "clementi": "clementi",
    "copland": "copland",
    "czerny": "czerny",
    "dvorak": "dvorak",
    "faure": "faure",
    "gershwin": "gershwin",
    "haendel": "haendel",
    "handel": "haendel",
    "holst": "holst",
    "hummel": "hummel",
    "joplin": "joplin",
    "massenet": "massenet",
    "ravel": "ravel",
    "rachmaninoff": "rachmaninoff",
    "rachmaninov": "rachmaninoff",
    "satie": "satie",
    "scriabin": "scriabin",
    "sibelius": "sibelius",
    "strauss": "strauss",
    "stravinsky": "stravinsky",
    "vivaldi": "vivaldi",
    "wagner": "wagner",
    "weber": "weber",
    "prokofiev": "prokofiev",
    "poulenc": "poulenc",
    "mussorgsky": "muss",
    "tchaikovsky": "tschai",
    "balakirev": "balakir",
    "granados": "granados",
    # Additional composers we'll accept
    "cpe_bach": "cpe_bach",
    "field": "field",
    "franck": "franck",
    "ginastera": "ginastera",
    "kuhlau": "kuhlau",
    "heller": "heller",
}


def download_dataset():
    """Download the HF dataset using huggingface_hub CLI."""
    print("Downloading dataset from HuggingFace...")
    os.makedirs(HF_DOWNLOAD_DIR, exist_ok=True)
    ret = os.system(
        f"hf download drengskapur/midi-classical-music "
        f"--repo-type dataset "
        f"--local-dir {HF_DOWNLOAD_DIR}"
    )
    if ret != 0:
        print("Failed to download. Trying with full path...")
        ret = os.system(
            f"/usr/bin/hf download drengskapur/midi-classical-music "
            f"--repo-type dataset "
            f"--local-dir {HF_DOWNLOAD_DIR}"
        )
    if ret != 0:
        print("ERROR: Failed to download dataset")
        sys.exit(1)
    print("Download complete!")


def organize_files():
    """Organize downloaded MIDI files into composer subdirectories."""
    data_dir = os.path.join(HF_DOWNLOAD_DIR, "data")
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    midi_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.mid'))
    print(f"\nFound {len(midi_files)} MIDI files in downloaded dataset")
    
    # Get existing composers
    existing_composers = set(d for d in os.listdir(MIDI_DIR) 
                            if os.path.isdir(os.path.join(MIDI_DIR, d)))
    print(f"Existing composers in midi_files/: {len(existing_composers)}")
    
    # Count existing files
    existing_count = sum(
        len([f for f in os.listdir(os.path.join(MIDI_DIR, c)) if f.endswith('.mid')])
        for c in existing_composers
    )
    print(f"Existing MIDI files: {existing_count}")
    
    # Process each file
    copied = 0
    skipped_no_match = 0
    skipped_exists = 0
    new_composers = set()
    unmatched_composers = set()
    
    for filename in midi_files:
        # Extract composer from filename: "composer-piece_name.mid"
        parts = filename.split("-", 1)
        if len(parts) < 2:
            skipped_no_match += 1
            continue
        
        hf_composer = parts[0].lower()
        
        # Map to our composer directory name
        if hf_composer in COMPOSER_MAP:
            our_composer = COMPOSER_MAP[hf_composer]
        elif hf_composer in existing_composers:
            our_composer = hf_composer
        else:
            unmatched_composers.add(hf_composer)
            skipped_no_match += 1
            continue
        
        # Create composer directory if it doesn't exist
        composer_dir = os.path.join(MIDI_DIR, our_composer)
        if not os.path.isdir(composer_dir):
            os.makedirs(composer_dir, exist_ok=True)
            new_composers.add(our_composer)
        
        # Copy file (use the piece name part as filename to avoid collisions)
        # Replace the "composer-" prefix with just the piece name
        piece_name = parts[1]
        # Handle nested directory names in the HF filenames (e.g., "bach-bwv001-_400_chorales-xxx.mid")
        # Keep the full name after the first dash to preserve uniqueness
        dest_name = filename  # keep original name for uniqueness
        dest_path = os.path.join(composer_dir, dest_name)
        
        if os.path.exists(dest_path):
            skipped_exists += 1
            continue
            
        src_path = os.path.join(data_dir, filename)
        shutil.copy2(src_path, dest_path)
        copied += 1
    
    print(f"\n--- Results ---")
    print(f"Files copied: {copied}")
    print(f"Skipped (already exists): {skipped_exists}")
    print(f"Skipped (no composer match): {skipped_no_match}")
    if new_composers:
        print(f"New composers added: {sorted(new_composers)}")
    if unmatched_composers:
        print(f"Unmatched composers (not added): {sorted(unmatched_composers)}")
    
    # Final count
    all_composers = sorted(d for d in os.listdir(MIDI_DIR) 
                          if os.path.isdir(os.path.join(MIDI_DIR, d)))
    total_files = sum(
        len([f for f in os.listdir(os.path.join(MIDI_DIR, c)) if f.endswith('.mid')])
        for c in all_composers
    )
    print(f"\nTotal composers: {len(all_composers)}")
    print(f"Total MIDI files: {total_files}")
    print(f"\nComposer breakdown:")
    for c in all_composers:
        cdir = os.path.join(MIDI_DIR, c)
        n = len([f for f in os.listdir(cdir) if f.endswith('.mid')])
        marker = " (NEW)" if c in new_composers else ""
        print(f"  {c:20s}: {n:4d} files{marker}")


if __name__ == "__main__":
    download_dataset()
    organize_files()
    print("\nDone! You'll need to:")
    print("1. Delete .midi_cache/ to force re-tokenization")  
    print("2. Run: uv run python prepare.py")
    print("3. Then retrain with the larger dataset")
