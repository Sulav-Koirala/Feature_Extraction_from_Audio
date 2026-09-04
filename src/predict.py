import sys
from pathlib import Path
import joblib
import numpy as np
from . import config
from .audio_io import load_audio_file
from .features import FEATURE_NAMES, extract_features
from .preprocess import preprocess


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{path.name}' not found in {path.parent}. Train first:\n"
            f"    python -m src.build_dataset\n"
            f"    python -m src.train"
        )
    return joblib.load(path)


def _predict_with_conf(bundle, x):
    model = bundle["model"]
    names = bundle.get("feature_names", FEATURE_NAMES)
    row = np.array([[x[n] for n in names]], dtype=np.float32)
    label = str(model.predict(row)[0])
    conf = float(model.predict_proba(row)[0].max()) if hasattr(model, "predict_proba") else None
    return label, conf


def predict_from_file(audio_path) -> dict:
    speed_bundle = load_model(config.SPEED_MODEL_PATH)
    loud_bundle = load_model(config.LOUDNESS_MODEL_PATH)

    y, sr = load_audio_file(audio_path)
    y, sr = preprocess(y, sr)
    if y.size == 0 or (len(y) / sr) < config.MIN_DURATION_SEC:
        raise ValueError("Audio is empty or too short after trimming silence.")

    feats = extract_features(y, sr)
    speed_label, speed_conf = _predict_with_conf(speed_bundle, feats)
    loud_label, loud_conf = _predict_with_conf(loud_bundle, feats)

    return {
        "features": feats,
        "speed": {"label": speed_label, "confidence": speed_conf},
        "loudness": {"label": loud_label, "confidence": loud_conf},
    }


def _print_report(audio_path, result):
    feats = result["features"]
    bar = "=" * 60
    print(bar)
    print(f"  AUDIO ANALYSIS:  {audio_path}")
    print(bar)

    print("\n  --- Extracted features ---")
    print(f"  Duration            : {feats['duration']:.2f} s")
    print(f"  RMS energy (mean)   : {feats['rms_mean']:.5f}")
    print(f"  RMS energy (max)    : {feats['rms_max']:.5f}")
    print(f"  Loudness (mean dB)  : {feats['rms_db_mean']:.2f} dB")
    print(f"  Zero-crossing rate  : {feats['zcr_mean']:.4f}")
    print(f"  Onset rate          : {feats['onset_rate']:.2f} onsets/s")
    print(f"  Tempo               : {feats['tempo']:.1f} BPM")
    print(f"  Spectral centroid   : {feats['spectral_centroid_mean']:.1f} Hz")
    mfcc_means = ", ".join(
        f"{feats[f'mfcc_{i}_mean']:.1f}" for i in range(1, config.N_MFCC + 1)
    )
    print(f"  MFCC means (1..{config.N_MFCC})   : [{mfcc_means}]")

    print("\n  --- Predictions ---")
    s, l = result["speed"], result["loudness"]
    s_conf = f"  (confidence {s['confidence'] * 100:.0f}%)" if s["confidence"] is not None else ""
    l_conf = f"  (confidence {l['confidence'] * 100:.0f}%)" if l["confidence"] is not None else ""
    print(f"  Speaking speed      : {s['label']}{s_conf}")
    print(f"  Loudness level      : {l['label']}{l_conf}")
    print(bar)


def ask_for_file() -> str:
    return input("Enter path to an audio file: ").strip()


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    audio_path = argv[0] if argv else ask_for_file()
    if not audio_path:
        print("No audio file provided.")
        return 1
    try:
        result = predict_from_file(audio_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Failed to analyze '{audio_path}': {exc}")
        return 1
    _print_report(audio_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
