from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from .dataset import gather_feature_paths, raw_feature_iterator
from .features import PEFeatureExtractor
from .labels import build_label_map, save_label_map


def _write_label_value(
    y_path: Path | str,
    irow: int,
    label,
    label_type: str,
    label_map: dict,
    nrows: int,
) -> None:
    if label_type in {"label", "family"}:
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows,))
        if label is None:
            y[irow] = -1
        elif isinstance(label, int):
            y[irow] = label
        elif isinstance(label, str):
            y[irow] = label_map.get(label, -1)
        else:
            raise ValueError(f"Unable to parse label format for {label_type}: {type(label)}")
        return

    y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows, len(label_map)))
    if label is None:
        return
    if not isinstance(label, list):
        raise ValueError(f"Expected list label for multilabel task {label_type}")
    for single_label in label:
        mapped = label_map.get(single_label)
        if mapped is not None:
            y[irow, mapped] = 1


def vectorize(
    irow: int,
    raw_features_string: str,
    x_path: Path | str,
    y_path: Path | str,
    extractor: PEFeatureExtractor,
    nrows: int,
    label_type: str = "label",
    label_map: dict | None = None,
) -> None:
    """
    Vectorize one sample and write it into memory-mapped X/y files.
    """
    if label_map is None:
        label_map = {}

    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError(f"Invalid label_type: {label_type}")

    x = np.memmap(x_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    x[irow] = feature_vector

    label = raw_features[label_type]
    _write_label_value(y_path, irow, label, label_type, label_map, nrows)


def _vectorize_unpack(args):
    return vectorize(*args)


def vectorize_subset(
    x_path: Path | str,
    y_path: Path | str,
    raw_feature_paths: list[Path],
    extractor: PEFeatureExtractor,
    nrows: int,
    label_type: str = "label",
    label_map: dict | None = None,
) -> None:
    """
    Vectorize a subset and write X/y to disk.
    """
    if label_map is None:
        label_map = {}

    x_path = Path(x_path)
    y_path = Path(y_path)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    y_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    del x

    if label_type in {"label", "family"}:
        y = np.memmap(y_path, dtype=np.int32, mode="w+", shape=(nrows,))
        y[:] = -1
    else:
        y = np.memmap(y_path, dtype=np.int32, mode="w+", shape=(nrows, len(label_map)))
        y[:] = 0
    del y

    ctx = mp.get_context("spawn")
    workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    argument_iterator = (
        (irow, raw_features_string, x_path, y_path, extractor, nrows, label_type, label_map)
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )

    with ctx.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_vectorize_unpack, argument_iterator, chunksize=32):
            pass


def create_vectorized_features(
    data_dir: Path | str,
    label_type: str = "label",
    class_min: int = 10,
    output_dir: Path | str | None = None,
    save_label_map_file: bool = True,
) -> dict[str, int]:
    """
    Create vectorized feature files for train / test / challenge.

    Returns:
        label_map for non-binary tasks, otherwise {}.
    """
    extractor = PEFeatureExtractor()
    data_path = Path(data_dir)
    out_path = Path(output_dir) if output_dir is not None else data_path
    out_path.mkdir(parents=True, exist_ok=True)

    train_paths = gather_feature_paths(data_path, "train")
    test_paths = gather_feature_paths(data_path, "test")
    challenge_paths = gather_feature_paths(data_path, "challenge")

    train_nrows = sum(1 for fp in train_paths for _ in fp.open("r", encoding="utf-8"))
    test_nrows = sum(1 for fp in test_paths for _ in fp.open("r", encoding="utf-8"))
    challenge_nrows = sum(1 for fp in challenge_paths for _ in fp.open("r", encoding="utf-8"))

    label_map = build_label_map(data_path, label_type, class_min=class_min)

    vectorize_subset(
        out_path / "X_train.dat",
        out_path / "y_train.dat",
        train_paths,
        extractor,
        train_nrows,
        label_type=label_type,
        label_map=label_map,
    )

    vectorize_subset(
        out_path / "X_test.dat",
        out_path / "y_test.dat",
        test_paths,
        extractor,
        test_nrows,
        label_type=label_type,
        label_map=label_map,
    )

    vectorize_subset(
        out_path / "X_challenge.dat",
        out_path / "y_challenge.dat",
        challenge_paths,
        extractor,
        challenge_nrows,
        label_type=label_type,
        label_map=label_map,
    )

    if label_type != "label" and save_label_map_file:
        save_label_map(out_path / f"{label_type}_label_map.json", label_map)

    return label_map


def create_project_vectorized_features(
    source_data_dir: Path | str,
    binary_out_dir: Path | str,
    family_out_dir: Path | str,
    family_class_min: int = 10,
) -> None:
    """
    Create the 2 datasets needed by your project:
    - layer 1: binary label
    - layer 2: family label
    """
    create_vectorized_features(
        data_dir=source_data_dir,
        label_type="label",
        output_dir=binary_out_dir,
        save_label_map_file=False,
    )
    create_vectorized_features(
        data_dir=source_data_dir,
        label_type="family",
        class_min=family_class_min,
        output_dir=family_out_dir,
        save_label_map_file=True,
    )
