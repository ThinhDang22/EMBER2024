from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pl = None

try:  # pragma: no cover - optional fallback
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional fallback
    pd = None

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
    """Yield raw feature strings from the input file paths."""
    for path in file_paths:
        with Path(path).open("r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    yield line


def count_jsonl_rows(file_paths: list[Path]) -> int:
    """Count JSONL rows quickly using buffered binary reads."""
    total = 0
    for path in file_paths:
        with Path(path).open("rb") as fin:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                total += chunk.count(b"\n")
    return total


def gather_feature_paths(
    data_dir: Path | str,
    subset: str,
    filetype: str | None = None,
    week: str | None = None,
) -> list[Path]:
    """Gather paths to raw metadata .jsonl files in the given data_dir."""
    data_dir = Path(data_dir)
    feature_paths: list[Path] = []

    for path in sorted(data_dir.glob("*.jsonl")):
        file_name = path.name
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(path)

    if not feature_paths:
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths


def _resolve_x_path(data_path: Path, subset: str) -> Path:
    direct = data_path / f"X_{subset}.dat"
    if direct.is_file():
        return direct

    ref_path = data_path / f"X_{subset}.path"
    if ref_path.is_file():
        ref = ref_path.read_text(encoding="utf-8").strip()
        if not ref:
            raise ValueError(f"Reference path file is empty: {ref_path}")
        resolved = Path(ref)
        if not resolved.is_absolute():
            resolved = (ref_path.parent / resolved).resolve()
        if resolved.is_file():
            return resolved
        raise ValueError(f"Referenced X file does not exist: {resolved}")

    raise ValueError(f"Invalid subset feature file: expected {direct} or {ref_path}")


def _read_y_shape(data_path: Path, subset: str) -> tuple[int, ...] | None:
    meta_path = data_path / f"y_{subset}.shape.json"
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    shape = payload.get("shape")
    if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
        return tuple(shape)
    return None


def read_vectorized_features(data_dir: Path | str, subset: str = "train") -> tuple[np.memmap, np.ndarray]:
    """Read vectorized features from .dat files or X_*.path references."""
    data_path = Path(data_dir)
    x_path = _resolve_x_path(data_path, subset)
    y_path = data_path / f"y_{subset}.dat"

    if not y_path.is_file():
        raise ValueError(f"Invalid subset label file: {y_path}")

    ndim = PEFeatureExtractor().dim
    x = np.memmap(x_path, dtype=np.float32, mode="r")
    if x.size % ndim != 0:
        raise ValueError(
            f"Feature file size is invalid: total elements = {x.size}, not divisible by feature dimension = {ndim}"
        )
    nrows = x.size // ndim
    x = x.reshape(nrows, ndim)

    y = np.memmap(y_path, dtype=np.int32, mode="r")
    y_shape = _read_y_shape(data_path, subset)
    if y_shape is not None:
        expected = int(np.prod(y_shape)) if y_shape else 0
        if expected != y.size:
            raise ValueError(
                f"Label file size mismatch: shape metadata says {y_shape} -> {expected} values, found {y.size}"
            )
        if len(y_shape) == 1:
            return x, y.reshape(y_shape)
        return x, np.asarray(y).reshape(y_shape)

    if y.size == nrows:
        return x, y
    if nrows > 0 and y.size % nrows == 0:
        return x, np.asarray(y).reshape(nrows, y.size // nrows)

    raise ValueError(
        f"Label file size does not match feature file: X has {nrows} samples but y has {y.size} elements"
    )


def read_metadata_record(raw_features_string: str) -> dict:
    """Decode a raw features string and return only the metadata fields."""
    all_data = json.loads(raw_features_string)
    return {k: all_data.get(k) for k in ORDERED_COLUMNS if k in all_data}


def _read_metadata_subset_polars(paths: list[Path], subset: str):
    if pl is None:
        return None

    scans = []
    for path in paths:
        scans.append(pl.scan_ndjson(str(path), infer_schema_length=1000))

    scan = pl.concat(scans, how="diagonal_relaxed") if len(scans) > 1 else scans[0]
    available = set(scan.collect_schema().names())
    cols = [c for c in ORDERED_COLUMNS if c in available]
    return scan.select(cols).with_columns(pl.lit(subset).alias("subset")).collect(streaming=True)


def _read_metadata_subset_pandas(paths: list[Path], subset: str):
    if pd is None:
        raise ModuleNotFoundError(
            "read_metadata requires either polars or pandas to be installed."
        )

    frames = []
    usecols = set(ORDERED_COLUMNS)
    for path in paths:
        for chunk in pd.read_json(path, lines=True, chunksize=10000):
            cols = [c for c in ORDERED_COLUMNS if c in chunk.columns]
            part = chunk.loc[:, cols].copy()
            part["subset"] = subset
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=ORDERED_COLUMNS + ["subset"])
    return pd.concat(frames, ignore_index=True)


def _read_metadata_subset(paths: list[Path], subset: str):
    polars_df = _read_metadata_subset_polars(paths, subset)
    if polars_df is not None:
        return polars_df
    return _read_metadata_subset_pandas(paths, subset)


def read_metadata(data_dir: Path | str):
    """
    Return metadata tables for train / test / challenge.

    Uses a streaming Polars path when available to reduce peak RAM,
    and falls back to chunked pandas reads otherwise.
    """
    data_path = Path(data_dir)
    return (
        _read_metadata_subset(gather_feature_paths(data_path, "train"), "train"),
        _read_metadata_subset(gather_feature_paths(data_path, "test"), "test"),
        _read_metadata_subset(gather_feature_paths(data_path, "challenge"), "challenge"),
    )
