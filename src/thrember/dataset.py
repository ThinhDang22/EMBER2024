from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from .features import PEFeatureExtractor


ORDERED_COLUMNS = [
    "sha256",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "label",
    "file_type",
    "family",
    "family_confidence",
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
]


def raw_feature_iterator(file_paths: list[Path]) -> Iterator[str]:
    """
    Yield raw feature strings from the input file paths.
    """
    for path in file_paths:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                yield line


def gather_feature_paths(
    data_dir: Path | str,
    subset: str,
    filetype: str | None = None,
    week: str | None = None,
) -> list[Path]:
    """
    Gather paths to raw metadata .jsonl files in the given data_dir.
    Supports filtering by subset, file type, and/or collection week.
    """
    data_dir = Path(data_dir)
    feature_paths: list[Path] = []

    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(data_dir / file_name)

    if not feature_paths:
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths


def read_vectorized_features(data_dir: Path | str, subset: str = "train") -> tuple[np.memmap, np.ndarray]:
    """
    Read vectorized features as memory-mapped arrays.

    X is returned as a memmap of shape (N, D).
    y is returned as either:
    - shape (N,) for binary / multiclass
    - shape (N, C) for multilabel
    """
    data_path = Path(data_dir)
    x_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not x_path.is_file():
        raise ValueError(f"Invalid subset file: {x_path}")
    if not y_path.is_file():
        raise ValueError(f"Invalid subset file: {y_path}")

    ndim = PEFeatureExtractor().dim

    x = np.memmap(x_path, dtype=np.float32, mode="r")
    if x.size % ndim != 0:
        raise ValueError(
            f"Feature file size is invalid: total elements = {x.size}, "
            f"not divisible by feature dimension = {ndim}"
        )
    nrows = x.size // ndim
    x = x.reshape(nrows, ndim)

    y = np.memmap(y_path, dtype=np.int32, mode="r")
    if y.size == nrows:
        return x, y
    if nrows > 0 and y.size % nrows == 0:
        y = y.reshape(nrows, y.size // nrows)
        return x, y

    raise ValueError(
        f"Label file size does not match feature file: "
        f"X has {nrows} samples but y has {y.size} elements"
    )


def read_metadata_record(raw_features_string: str) -> dict:
    """
    Decode a raw features string and return only the metadata fields.
    """
    all_data = json.loads(raw_features_string)
    return {k: all_data.get(k) for k in ORDERED_COLUMNS if k in all_data}


def _read_metadata_records(paths: list[Path]) -> list[dict]:
    ctx = mp.get_context("spawn")
    workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    with ctx.Pool(processes=workers) as pool:
        return list(pool.imap(read_metadata_record, raw_feature_iterator(paths)))


def read_metadata(data_dir: Path | str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Return metadata dataframes for train / test / challenge.
    """
    data_path = Path(data_dir)

    train_feature_paths = gather_feature_paths(data_path, "train")
    train_records = _read_metadata_records(train_feature_paths)
    train_columns = [c for c in ORDERED_COLUMNS if c in pl.DataFrame(train_records).columns]
    train_metadf = (
        pl.DataFrame(train_records)
        .with_columns(subset=pl.lit("train"))
        .select(train_columns + ["subset"])
    )

    test_feature_paths = gather_feature_paths(data_path, "test")
    test_records = _read_metadata_records(test_feature_paths)
    test_columns = [c for c in ORDERED_COLUMNS if c in pl.DataFrame(test_records).columns]
    test_metadf = (
        pl.DataFrame(test_records)
        .with_columns(subset=pl.lit("test"))
        .select(test_columns + ["subset"])
    )

    challenge_feature_paths = gather_feature_paths(data_path, "challenge")
    challenge_records = _read_metadata_records(challenge_feature_paths)
    challenge_columns = [c for c in ORDERED_COLUMNS if c in pl.DataFrame(challenge_records).columns]
    challenge_metadf = (
        pl.DataFrame(challenge_records)
        .with_columns(subset=pl.lit("challenge"))
        .select(challenge_columns + ["subset"])
    )

    return train_metadf, test_metadf, challenge_metadf
