"""Preprocessing: turn a raw ``(signal, sr)`` into a clean, standardized mono signal.

Order (roadmap Step 3): down-mix to mono -> resample to a common rate -> trim
leading/trailing silence.

We deliberately DO NOT amplitude-normalize: normalizing would erase the loudness
information we later measure (RMS/dB), which is exactly what the loudness label
depends on.
"""
from __future__ import annotations

import librosa
import numpy as np

from .config import SAMPLE_RATE, TRIM_TOP_DB


def to_mono(y: np.ndarray) -> np.ndarray:
    """Average channels down to a single mono channel.

    Expects soundfile's layout: 1-D (already mono) or 2-D ``(frames, channels)``.
    """
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    return y.mean(axis=1).astype(np.float32)


def resample_to(y: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Resample to ``target_sr`` (no-op if already at the target rate)."""
    if sr == target_sr:
        return y.astype(np.float32, copy=False), sr
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr


def trim_silence(y: np.ndarray, top_db: int = TRIM_TOP_DB) -> np.ndarray:
    """Trim leading/trailing silence; keep the original if trimming empties it."""
    if y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def preprocess(y: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Full mono -> resample -> trim pipeline. Returns ``(clean_signal, target_sr)``."""
    y = to_mono(y)
    y, sr = resample_to(y, sr, target_sr)
    y = trim_silence(y)
    return y, sr
