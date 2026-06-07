# -*- encoding: utf-8 -*-

# ===================================
# LTBio - LongTermBiosignals
#
# Script: serialize_adresso21_dataset
# Description: Loads every .wav recording in the ADReSSo21 dataset into a Speech
# biosignal object, serializes each one as '{patient_code}.biosignal', and zips
# all the resulting files into a single archive to be shared with the supervisor.
#
# Usage:
#   python scripts/serialize_adresso21_dataset.py
#
# Contributors: Seyedali Divbandroudbaraki
# Created: 07/06/2026
# ===================================

import os
import zipfile
from pathlib import Path

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21


# Resolve paths relative to the project root, regardless of the current working
# directory the script is launched from (e.g. PyCharm may run it from scripts/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / 'resources' / 'ADReSSO21_tests'        # root of the dataset
OUTPUT_DIR = PROJECT_ROOT / 'resources' / 'ADReSSo21_biosignals'    # where .biosignal files are saved
ZIP_PATH = PROJECT_ROOT / 'resources' / 'ADReSSo21_biosignals.zip'


def find_all_wav_files(dataset_dir: Path):
    """Recursively find every .wav file under the dataset's audio/ folders."""
    return sorted(dataset_dir.rglob('*.wav'))


def serialize_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = find_all_wav_files(DATASET_DIR)
    print(f"Found {len(wav_files)} .wav files under {DATASET_DIR}")

    succeeded, failed, skipped = [], [], []

    for filepath in wav_files:
        try:
            speech = Speech(str(filepath), ADReSSo21)
            code = speech.patient_code  # public property on Biosignal

            out_path = OUTPUT_DIR / f"{code}.biosignal"

            if out_path.exists():
                print(f"  [SKIP] {filepath.name} -> {out_path.name} (already exists, possible code collision)")
                skipped.append((filepath, code))
                continue

            speech.save(str(out_path))
            print(f"  [OK]   {filepath.name} -> {out_path.name}")
            succeeded.append((filepath, code))

        except Exception as e:
            print(f"  [FAIL] {filepath.name}: {type(e).__name__}: {e}")
            failed.append((filepath, str(e)))

    print()
    print(f"Done. Succeeded: {len(succeeded)} | Failed: {len(failed)} | Skipped (collisions): {len(skipped)}")

    if failed:
        print("\nFailed files:")
        for filepath, err in failed:
            print(f"  - {filepath}: {err}")

    if skipped:
        print("\nSkipped files (code collisions):")
        for filepath, code in skipped:
            print(f"  - {filepath} (code={code})")

    return succeeded


def zip_biosignal_files():
    biosignal_files = sorted(OUTPUT_DIR.glob('*.biosignal'))
    print(f"\nZipping {len(biosignal_files)} .biosignal files into {ZIP_PATH} ...")

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in biosignal_files:
            zf.write(f, arcname=f.name)

    print(f"Done. Archive size: {os.path.getsize(ZIP_PATH) / (1024 * 1024):.1f} MB")


if __name__ == '__main__':
    serialize_dataset()
    zip_biosignal_files()
