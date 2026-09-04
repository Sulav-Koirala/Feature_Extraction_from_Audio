"""Steps 4-5: extract features for every clip and attach self-generated labels.

Produces:
  * ``outputs/features.csv`` -- one row per clip (id, ~37 features, speed_score,
    speed_label, loudness_label)
  * ``outputs/metrics.json`` -- the percentile thresholds used for labeling

Run:
    python -m src.build_dataset            # all pooled clips (~780)
    python -m src.build_dataset --limit 60 # quick sample
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import config
from .features import FEATURE_NAMES, extract_features
from .labeling import (
    assign_loudness,
    assign_speed,
    compute_speed_score,
    percentile_thresholds,
)
from .preprocess import preprocess


def build_feature_table(limit=None, files=None, verbose=True) -> pd.DataFrame:
    """Iterate clips -> preprocess -> extract features -> DataFrame (no labels yet)."""
    from .data import iter_clips

    rows = []
    skipped = 0
    t0 = time.time()
    for n, (row_id, y, sr) in enumerate(iter_clips(files=files, limit=limit), start=1):
        y, sr = preprocess(y, sr)
        duration = len(y) / sr if sr else 0.0
        if duration < config.MIN_DURATION_SEC:
            skipped += 1
            continue
        feats = extract_features(y, sr)
        feats["id"] = int(row_id)
        rows.append(feats)
        if verbose and n % 50 == 0:
            print(f"  processed {n} clips ({time.time() - t0:.1f}s)...")

    if not rows:
        raise RuntimeError("No clips were processed -- check dataset download / decoding.")
    if verbose:
        print(f"Extracted features for {len(rows)} clips (skipped {skipped} too-short).")

    df = pd.DataFrame(rows)
    return df[["id"] + FEATURE_NAMES]  # id first, then canonical feature order


def add_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach ``speed_label`` and ``loudness_label`` via percentile bucketing."""
    df = df.copy()

    # Loudness from rms_db_mean.
    loud_lo, loud_hi = percentile_thresholds(df["rms_db_mean"].to_numpy())
    df["loudness_label"] = df["rms_db_mean"].apply(
        lambda v: assign_loudness(v, (loud_lo, loud_hi))
    )

    # Speed from a blend of zcr_mean + onset_rate.
    speed_score = compute_speed_score(df["zcr_mean"].to_numpy(), df["onset_rate"].to_numpy())
    df["speed_score"] = speed_score
    speed_lo, speed_hi = percentile_thresholds(speed_score)
    df["speed_label"] = df["speed_score"].apply(
        lambda v: assign_speed(v, (speed_lo, speed_hi))
    )

    thresholds = {
        "loudness": {"lower": loud_lo, "upper": loud_hi, "source": "rms_db_mean"},
        "speed": {
            "lower": speed_lo,
            "upper": speed_hi,
            "source": "mean(zscore(zcr_mean), zscore(onset_rate))",
        },
    }
    return df, thresholds


def main(limit=None, files=None):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building feature table from dataset '{config.HF_DATASET_ID}' ...")

    df = build_feature_table(limit=limit, files=files)
    df, thresholds = add_labels(df)

    df.to_csv(config.FEATURES_CSV, index=False)
    print(f"\nSaved feature table -> {config.FEATURES_CSV}  ({len(df)} rows)")

    print("\nLabel distribution (sanity check -- thirds should be roughly balanced):")
    print("  speed:   ", df["speed_label"].value_counts().to_dict())
    print("  loudness:", df["loudness_label"].value_counts().to_dict())

    # Persist thresholds (merge into metrics.json if it already exists).
    metrics = {}
    if config.METRICS_JSON.exists():
        try:
            metrics = json.loads(config.METRICS_JSON.read_text())
        except Exception:
            metrics = {}
    metrics["label_thresholds"] = thresholds
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Saved label thresholds -> {config.METRICS_JSON}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the feature table + labels.")
    parser.add_argument("--limit", type=int, default=None, help="cap number of clips")
    args = parser.parse_args()
    main(limit=args.limit)
