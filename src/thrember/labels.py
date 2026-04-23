from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .dataset import gather_feature_paths, raw_feature_iterator


IGNORE_TAGS = {"", "win32", "win64", "elf", "linux", "pdf", "apk", "android"}


def read_label(raw_features_string: str, label_type: str):
    """Read the target field from one raw feature line."""
    raw_features = json.loads(raw_features_string)
    return raw_features.get(label_type)


def read_label_subset(raw_feature_paths: list[Path], label_type: str) -> dict[str, int]:
    """Count unique labels/tags in a subset using a streaming pass."""
    counts: Counter[str] = Counter()
    for raw_features_string in raw_feature_iterator(raw_feature_paths):
        labels = read_label(raw_features_string, label_type)
        if labels is None:
            continue
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            if isinstance(label, str):
                counts[label] += 1
    return dict(counts)


def build_label_map(data_dir: Path | str, label_type: str, class_min: int = 10) -> dict[str, int]:
    """Build a numeric label map from the training split only."""
    if label_type == "label":
        return {}

    data_path = Path(data_dir)
    feature_paths = gather_feature_paths(data_path, "train")
    label_counts = read_label_subset(feature_paths, label_type)

    label_map: dict[str, int] = {}
    next_id = 0
    for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if label in IGNORE_TAGS:
            continue
        if count < class_min:
            continue
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
