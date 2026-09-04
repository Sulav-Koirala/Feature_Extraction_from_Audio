"""Central configuration: paths, audio parameters, and model/labeling settings.

Everything tunable lives here so the rest of the code has no scattered magic
numbers. Paths are resolved relative to this file (never hard-coded to one
machine), so the project runs cleanly from a fresh checkout.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFUSION_DIR = OUTPUTS_DIR / "confusion_matrices"

FEATURES_CSV = OUTPUTS_DIR / "features.csv"
METRICS_JSON = OUTPUTS_DIR / "metrics.json"
SPEED_MODEL_PATH = MODELS_DIR / "speed_model.pkl"
LOUDNESS_MODEL_PATH = MODELS_DIR / "loudness_model.pkl"

# ---------------------------------------------------------------------------
# Dataset (Hugging Face) -- public, Apache-2.0, WAV audio @ 16 kHz.
# We read the parquet files directly (see src/data.py) rather than using the
# `datasets` library, to avoid its torchcodec/ffmpeg audio-decoding path.
# ---------------------------------------------------------------------------
HF_DATASET_ID = "devrahulbanjara/ne-en-codeswitching-asr-technical-interview"
HF_PARQUET_FILES = [
    "data/train-00000-of-00001.parquet",       # ~639 clips
    "data/validation-00000-of-00001.parquet",  # smallest split (good for a quick sample)
    "data/test-00000-of-00001.parquet",
]

# ---------------------------------------------------------------------------
# Audio / feature-extraction parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000        # target sampling rate (dataset is already 16 kHz)
N_MFCC = 13                # number of MFCC coefficients
TRIM_TOP_DB = 30           # silence-trim threshold, in dB below the peak
MIN_DURATION_SEC = 0.5     # clips shorter than this (after trimming) are skipped

# ---------------------------------------------------------------------------
# Labeling / training parameters
# ---------------------------------------------------------------------------
LOWER_PERCENTILE = 100.0 / 3.0   # ~33.3 -> Soft/Slow  vs Normal boundary
UPPER_PERCENTILE = 200.0 / 3.0   # ~66.7 -> Normal     vs Loud/Fast boundary
TEST_SIZE = 0.20
SEED = 42

# Class order (kept consistent everywhere: quietest/slowest -> loudest/fastest)
LOUDNESS_CLASSES = ["Soft", "Normal", "Loud"]
SPEED_CLASSES = ["Slow", "Normal", "Fast"]
