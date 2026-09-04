"""Feature extraction -- the single source of truth for the model's input vector.

``extract_features`` is used by BOTH dataset building and prediction, so the
feature order can never drift between training and inference. ``FEATURE_NAMES``
is that fixed, canonical order; every consumer trusts it.

Feature groups:
  * MFCC mean/std   -> overall tone / timbre ("what the voice sounds like")
  * RMS energy + dB -> loudness (drives the Soft/Normal/Loud label)
  * Zero-crossing   -> how "busy" the waveform is (speed proxy)
  * Spectral cent.  -> brightness (extra context)
  * Onset rate      -> syllable-like pulses per second (primary speed proxy)
  * Tempo, duration -> supporting context
"""
from __future__ import annotations

from collections import OrderedDict

import librosa
import numpy as np

from .config import N_MFCC, SAMPLE_RATE

# Canonical feature order. Anything reading a CSV / model bundle relies on this.
FEATURE_NAMES: list[str] = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "rms_std", "rms_max", "rms_db_mean"]
    + ["zcr_mean", "zcr_std"]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["onset_rate", "tempo", "duration"]
)


def _estimate_tempo(y: np.ndarray, sr: int) -> float:
    """Tempo (BPM) via beat tracking; defensive so a failure never breaks a clip."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(np.atleast_1d(tempo)[0])
    except Exception:
        return 0.0


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> "OrderedDict[str, float]":
    """Compute the fixed-length feature dictionary for one clip.

    Returns an ``OrderedDict`` keyed exactly by ``FEATURE_NAMES``.
    """
    feats: OrderedDict[str, float] = OrderedDict()

    duration = float(len(y)) / float(sr) if sr else 0.0

    # Guard: empty signal -> all-zero vector (keeps the pipeline from crashing).
    if y.size == 0:
        for name in FEATURE_NAMES:
            feats[name] = 0.0
        return feats

    # --- MFCC (timbre / general tone shape) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # shape (N_MFCC, frames)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc_mean[i])
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_std"] = float(mfcc_std[i])

    # --- RMS energy (loudness) ---
    rms = librosa.feature.rms(y=y)[0]  # shape (frames,)
    rms_mean = float(rms.mean())
    feats["rms_mean"] = rms_mean
    feats["rms_std"] = float(rms.std())
    feats["rms_max"] = float(rms.max())
    # Convert mean RMS to a dB-like value; +eps avoids log(0). This is the
    # canonical loudness measure used to generate the Soft/Normal/Loud labels.
    feats["rms_db_mean"] = float(20.0 * np.log10(rms_mean + 1e-10))

    # --- Zero-crossing rate (speed proxy) ---
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())

    # --- Spectral centroid (brightness) ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(cent.mean())
    feats["spectral_centroid_std"] = float(cent.std())

    # --- Onset rate (primary speed proxy) + tempo ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    feats["onset_rate"] = float(len(onsets) / duration) if duration > 0 else 0.0
    feats["tempo"] = _estimate_tempo(y, sr)
    feats["duration"] = duration

    return feats


def features_to_vector(feats: dict) -> np.ndarray:
    """Turn a feature dict into a 1-D array in the canonical ``FEATURE_NAMES`` order."""
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
