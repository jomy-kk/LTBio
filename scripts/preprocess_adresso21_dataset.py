# -*- encoding: utf-8 -*-

# ===================================
# LTBio - LongTermBiosignals
#
# Script: preprocess_adresso21_dataset
# Description: Loads every '.biosignal' file produced for the ADReSSo21 dataset,
# runs each one through a Pre-processing II pipeline (filter, normalize, re-arrange),
# and saves the result as '{patient_code}.biosignal' in a separate output folder.
#
# Pipeline (see project roadmap "Plan 07.06.2026", phase "Pre-processing II"):
#   1. Filter    - FrequencyDomainFilter (band-pass, speech frequency range)
#   2. Normalize - Normalizer (z-score / zero mean & unit variance)
#   3. Re-arrange - resample to a common sampling frequency (16 kHz, via Biosignal.resample)
#
# Usage:
#   python scripts/preprocess_adresso21_dataset.py
#
# Contributors: Seyedali Divbandroudbaraki
# Created: 07/06/2026
# ===================================

from pathlib import Path

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.pipeline.Pipeline import Pipeline
from ltbio.processing.filters.FrequencyDomainFilter import FrequencyDomainFilter, FrequencyResponse, BandType
from ltbio.processing.formaters.Normalizer import Normalizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / 'resources' / 'ADReSSo21_biosignals'              # '.biosignal' files
OUTPUT_DIR = PROJECT_ROOT / 'resources' / 'ADReSSo21_biosignals_preprocessed'  # where pre-processed '.biosignal' files are saved

# Re-arrange: common sampling frequency to resample every recording to.
# 16 kHz is the standard input rate expected by pre-trained speech models
# (e.g. wav2vec 2.0 / HuBERT - see roadmap phase "ML Routine"), and the
# ADReSSo21 recordings are originally sampled at 44.1 kHz.
TARGET_SAMPLING_FREQUENCY = 16000.


def build_pipeline() -> Pipeline:
    """
    Builds the Pre-processing II pipeline (filter -> normalize), reusing LTBio's
    existing Visitor (Filter) and Composite (Pipeline/PipelineUnit) patterns.

    Steps:
      1. Band-pass filter (Butterworth, 80-8000 Hz) - keeps the human speech band,
         removes low-frequency rumble and high-frequency noise.
      2. Normalizer ('mean' / z-score) - zero mean & unit variance, the same
         normalization wav2vec 2.0 expects of its raw waveform input
         (Baevski et al., 2020, Section 2: "The raw waveform input to the
         encoder is normalized to zero mean and unit variance.").
    """
    pipeline = Pipeline(name='ADReSSo21 Pre-processing II')
    pipeline.add(FrequencyDomainFilter(FrequencyResponse.BUTTER, BandType.BANDPASS, cutoff=(80., 8000.), order=4))
    pipeline.add(Normalizer(method='mean'))
    return pipeline


def find_all_biosignal_files(input_dir: Path):
    """Find every '.biosignal' file produced by the serialization step."""
    return sorted(input_dir.glob('*.biosignal'))


def preprocess_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    biosignal_files = find_all_biosignal_files(INPUT_DIR)
    print(f"Found {len(biosignal_files)} .biosignal files under {INPUT_DIR}")

    pipeline = build_pipeline()
    print(f"Pipeline: {pipeline}")

    succeeded, failed, skipped = [], [], []

    for filepath in biosignal_files:
        try:
            out_path = OUTPUT_DIR / filepath.name  # same '{patient_code}.biosignal' name

            if out_path.exists():
                print(f"  [SKIP] {filepath.name} (already exists)")
                skipped.append(filepath)
                continue

            speech = Speech.load(str(filepath))

            # Step 1 & 2: filter (band-pass) and normalize (z-score), via the Pipeline.
            # Pipeline.__call__ returns a NEW Biosignal (prototype/'_new' pattern);
            # the loaded 'speech' is left untouched.
            processed = pipeline(speech)

            # Step 3: re-arrange - resample every channel to a common sampling
            # frequency. Unlike filter/normalize, there is no Resampler
            # PipelineUnit in LTBio, so this is called directly on the Biosignal
            # (Biosignal.resample mutates 'processed' in place; 'speech', the
            # originally loaded biosignal, is still left untouched).
            processed.resample(TARGET_SAMPLING_FREQUENCY)

            processed.save(str(out_path))
            print(f"  [OK]   {filepath.name} -> filtered, normalized & resampled to {int(TARGET_SAMPLING_FREQUENCY)} Hz")
            succeeded.append(filepath)

        except Exception as e:
            print(f"  [FAIL] {filepath.name}: {type(e).__name__}: {e}")
            failed.append((filepath, str(e)))

    print()
    print(f"Done. Succeeded: {len(succeeded)} | Failed: {len(failed)} | Skipped: {len(skipped)}")

    if failed:
        print("\nFailed files:")
        for filepath, err in failed:
            print(f"  - {filepath}: {err}")

    return succeeded


if __name__ == '__main__':
    preprocess_dataset()
