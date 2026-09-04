"""Self-generated 3-class labels from signal measurements (roadmap Step 5).

The dataset has no fast/normal/slow or loud/normal/soft labels, so we create them
by bucketing clips relative to the whole dataset:

  * loudness  <- ``rms_db_mean``
  * speed     <- a blend of ``zcr_mean`` and ``onset_rate`` (each z-scored,
                 then averaged into one "busyness" score)

The 33rd and 66th percentiles split each measure into three roughly equal thirds.
These labels are heuristic (not human-annotated) -- an honest limitation of the
approach, exactly as the roadmap notes.
"""
from __future__ import annotations

import numpy as np

from .config import (
    LOUDNESS_CLASSES,
    LOWER_PERCENTILE,
    SPEED_CLASSES,
    UPPER_PERCENTILE,
)


def percentile_thresholds(values) -> tuple[float, float]:
    """Return the (lower, upper) percentile cut-offs for a list of values."""
    arr = np.asarray(values, dtype=float)
    lo = float(np.percentile(arr, LOWER_PERCENTILE))
    hi = float(np.percentile(arr, UPPER_PERCENTILE))
    return lo, hi


def _bucket(value: float, lo: float, hi: float, classes: list[str]) -> str:
    if value < lo:
        return classes[0]
    if value < hi:
        return classes[1]
    return classes[2]


def assign_loudness(value: float, thresholds: tuple[float, float]) -> str:
    """Map an ``rms_db_mean`` value to Soft / Normal / Loud."""
    lo, hi = thresholds
    return _bucket(value, lo, hi, LOUDNESS_CLASSES)


def assign_speed(value: float, thresholds: tuple[float, float]) -> str:
    """Map a speed-score value to Slow / Normal / Fast."""
    lo, hi = thresholds
    return _bucket(value, lo, hi, SPEED_CLASSES)


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def compute_speed_score(zcr_mean, onset_rate) -> np.ndarray:
    """Blend ZCR and onset-rate into one 'busyness' score (mean of z-scores).

    Combining two proxies is more robust than either alone (ZCR reacts to pitch
    and consonants, onset-rate to syllable pulses).
    """
    z = _zscore(np.asarray(zcr_mean, dtype=float))
    o = _zscore(np.asarray(onset_rate, dtype=float))
    return (z + o) / 2.0
