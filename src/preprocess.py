import librosa
import numpy as np
from .config import SAMPLE_RATE, TRIM_TOP_DB


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    return y.mean(axis=1).astype(np.float32)


def resample_to(y: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return y.astype(np.float32, copy=False), sr
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr


def trim_silence(y: np.ndarray, top_db: int = TRIM_TOP_DB) -> np.ndarray:
    if y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def preprocess(y: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    y = to_mono(y)
    y, sr = resample_to(y, sr, target_sr)
    y = trim_silence(y)
    return y, sr
