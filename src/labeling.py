import numpy as np

from .config import LOUDNESS_CLASSES, LOWER_PERCENTILE, SPEED_CLASSES, UPPER_PERCENTILE


def percentile_thresholds(values) -> tuple[float, float]:
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
    lo, hi = thresholds
    return _bucket(value, lo, hi, LOUDNESS_CLASSES)


def assign_speed(value: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    return _bucket(value, lo, hi, SPEED_CLASSES)


def compute_speed_score(onset_rate) -> np.ndarray:
    return np.asarray(onset_rate, dtype=float)
