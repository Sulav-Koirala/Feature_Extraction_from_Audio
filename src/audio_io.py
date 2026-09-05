import io
from pathlib import Path
import numpy as np
import soundfile as sf


def decode_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return y, int(sr)


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return y, int(sr)
    except Exception as exc: 
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=None, mono=False)
            y = np.asarray(y, dtype="float32")
            if y.ndim == 2:  
                y = y.T
            return y, int(sr)
        except Exception:
            raise RuntimeError(
                f"Could not decode audio file '{path}'. Original error: {exc}"
            ) from exc
