from typing import Iterator
import numpy as np
from .audio_io import decode_wav_bytes
from .config import HF_DATASET_ID, HF_PARQUET_FILES


def download_parquets(files: list[str] | None = None) -> list[str]:
    from huggingface_hub import hf_hub_download

    files = files or HF_PARQUET_FILES
    paths = []
    for fname in files:
        local = hf_hub_download(repo_id=HF_DATASET_ID, filename=fname, repo_type="dataset")
        paths.append(local)
    return paths


def _extract_bytes(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    try:  
        return dict(cell).get("bytes")
    except Exception:
        return None


def iter_clips(files: list[str] | None = None, limit: int | None = None) -> Iterator[tuple[int, np.ndarray, int]]:
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
                    continue  
                yield int(row_id), y, sr
                yielded += 1
