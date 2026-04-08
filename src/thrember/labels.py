from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

from .dataset import gather_feature_paths, raw_feature_iterator


IGNORE_TAGS = {"", "win32", "win64", "elf", "linux", "pdf", "apk", "android"}


def read_label(raw_features_string: str, label_type: str):
    """
    Read the target field from one raw feature line.
    """
    raw_features = json.loads(raw_features_string)
    return raw_features.get(label_type)


def _read_label_unpack(args):
    return read_label(*args)


def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> dict:
    """
    Count unique labels/tags in a subset.
    """
    ctx = mp.get_context("spawn")
    workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    argument_iterator = (
        (raw_features_string, label_type)
        for raw_features_string in raw_feature_iterator(raw_feature_paths)
    )

    label_counts: dict = {}
    with ctx.Pool(processes=workers) as pool:
        for labels in pool.imap_unordered(_read_label_unpack, argument_iterator, chunksize=64):
            if labels is None:
                continue
            if not isinstance(labels, list):
                labels = [labels]
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

    return label_counts


def build_label_map(data_dir: Path | str, label_type: str, class_min: int = 10) -> dict[str, int]:
    """
    Build a numeric label map for multiclass/multilabel tasks.
    """
    if label_type == "label":
        return {}

    data_path = Path(data_dir)
    label_map: dict[str, int] = {}
    next_id = 0

    for subset in ["train", "test"]:
        feature_paths = gather_feature_paths(data_path, subset)
        nrows = sum(1 for fp in feature_paths for _ in fp.open("r", encoding="utf-8"))
        label_counts = read_label_subset(feature_paths, nrows, label_type)

        for label, count in label_counts.items():
            if label in IGNORE_TAGS:
                continue
            if label in label_map:
                continue
            if count >= class_min:
                label_map[label] = next_id
                next_id += 1

    return label_map


def save_label_map(path: Path | str, label_map: dict[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(label_map, fout, indent=2, ensure_ascii=False)


def load_label_map(path: Path | str) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as fin:
        return json.load(fin)
