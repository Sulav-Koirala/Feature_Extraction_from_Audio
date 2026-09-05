# Audio Feature Extraction: Speed & Loudness Prediction

Given an audio file, this extracts acoustic features (MFCC, RMS energy/dB, ZCR,
onset-rate/tempo, spectral centroid) and predicts two things:

- **Speaking speed:** Fast / Normal / Slow
- **Loudness level:** Loud / Normal / Soft

Labels are self-generated from the dataset via percentile bucketing (the dataset
has no such labels), then two Random Forest classifiers are trained. Everything
runs from the command line, no web UI.

## Setup

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Run the pipeline

```bash
.venv/bin/python -m src.build_dataset

.venv/bin/python -m src.train

.venv/bin/python -m src.predict path/to/audio.wav
```

`build_dataset` accepts `--limit N` to process only the first N clips (quick test).


## Layout

```
src/
  config.py         constants (sample rate, paths, seeds, thresholds)
  audio_io.py       decode WAV bytes / load audio files (soundfile)
  preprocess.py     mono → resample → trim silence (no amplitude normalization)
  features.py       extract_features(): the single source of truth for the model input
  labeling.py       percentile-based Soft/Normal/Loud and Slow/Normal/Fast labels
  data.py           download + stream the HF dataset parquet files
  build_dataset.py  Steps 4–5: features + labels → outputs/features.csv
  train.py          Steps 6–8: train two models, evaluate, save
  predict.py        Steps 9–10: CLI predictor
models/             trained models (created by src.train)
outputs/            features.csv, metrics.json, confusion matrices
```

## Notes

- Dataset: [`devrahulbanjara/ne-en-codeswitching-asr-technical-interview`](https://huggingface.co/datasets/devrahulbanjara/ne-en-codeswitching-asr-technical-interview)
  (public, WAV @ 16 kHz). It's cached under `~/.cache/huggingface`, no local `data/` folder.
- Feature extraction is language-agnostic, so the Nepali–English mix is irrelevant.
- Because labels are derived from a subset of the features, accuracy is high by
  design; `src.train` prints train vs. test vs. cross-val accuracy so you can
  confirm the small gap = no overfitting. Labels are heuristic, not human-annotated.