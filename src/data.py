"""Download the dataset's parquet files and iterate over decoded audio clips.

We fetch the parquet files with ``huggingface_hub`` (cached under
``~/.cache/huggingface``) and stream them with ``pyarrow`` -- deliberately NOT
the ``datasets`` library, whose audio-decoding path can require torchcodec/ffmpeg
that aren't available on this setup.

Audio is stored as WAV bytes inside a struct column ``{bytes, path}``; we pull
the bytes out and decode them with soundfile.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .audio_io import decode_wav_bytes
from .config import HF_DATASET_ID, HF_PARQUET_FILES


def download_parquets(files: list[str] | None = None) -> list[str]:
    """Download the given parquet files from the HF hub; return local cached paths."""
    from huggingface_hub import hf_hub_download

    files = files or HF_PARQUET_FILES
    paths = []
    for fname in files:
        local = hf_hub_download(repo_id=HF_DATASET_ID, filename=fname, repo_type="dataset")
        paths.append(local)
    return paths


def _extract_bytes(cell) -> bytes | None:
    """Pull raw WAV bytes out of a parquet 'audio' cell.

    HF Audio columns serialize to a struct ``{'bytes': <binary>, 'path': <str>}``.
    """
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    try:  # pyarrow may hand back a struct scalar
        return dict(cell).get("bytes")
    except Exception:
        return None


def iter_clips(
    files: list[str] | None = None, limit: int | None = None
) -> Iterator[tuple[int, np.ndarray, int]]:
    """Yield ``(row_id, raw_signal, sample_rate)`` for clips across the parquet files.

    Streams in small batches to keep memory low. ``limit`` caps the total number
    of clips (used by the notebook for a quick sample). Audio is decoded but NOT
    preprocessed here -- callers do that.
    """
    import pyarrow.parquet as pq

    paths = download_parquets(files)
    yielded = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=64, columns=["id", "audio"]):
            ids = batch.column("id").to_pylist()
            audios = batch.column("audio").to_pylist()
            for row_id, cell in zip(ids, audios):
                if limit is not None and yielded >= limit:
                    return
                raw = _extract_bytes(cell)
                if not raw:
                    continue
                try:
                    y, sr = decode_wav_bytes(raw)
                except Exception:
                    continue  # skip undecodable rows rather than abort the run
                yield int(row_id), y, sr
                yielded += 1
