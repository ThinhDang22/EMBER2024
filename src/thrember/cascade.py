from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import read_vectorized_features
from .modeling import UnifiedModel


@dataclass
class CascadeClassifier:
    """Layer 1: benign vs malware; layer 2: family classification."""

    layer1: UnifiedModel
    layer2: UnifiedModel
    threshold: float = 0.5
    family_label_map: dict[str, int] | None = None

    def __post_init__(self):
        self.inverse_family_map = None
        if self.family_label_map:
            self.inverse_family_map = {v: k for k, v in self.family_label_map.items()}

    def predict_one(self, x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        malware_score = float(np.asarray(self.layer1.predict_scores(x))[0])
        if malware_score < self.threshold:
            return {"is_malware": 0, "malware_score": malware_score, "family_id": None, "family_name": None}

        family_id = int(self.layer2.predict_labels(x)[0])
        family_name = self.inverse_family_map.get(family_id) if self.inverse_family_map is not None else None
        return {"is_malware": 1, "malware_score": malware_score, "family_id": family_id, "family_name": family_name}

    def predict_batch(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Batch prediction — vectorized, không gọi predict_one per row."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Layer 1: lấy malware score toàn bộ batch một lần.
        scores = np.asarray(self.layer1.predict_scores(x), dtype=np.float64).ravel()
        is_malware = (scores >= self.threshold).astype(np.int32)

        # Layer 2: chỉ predict các row được phân loại là malware.
        family_ids: list[int | None] = [None] * len(x)
        malware_idx = np.where(is_malware == 1)[0]
        if len(malware_idx) > 0:
            preds = self.layer2.predict_labels(x[malware_idx])
            for local_i, global_i in enumerate(malware_idx):
                family_ids[global_i] = int(preds[local_i])

        results = []
        for i in range(len(x)):
            fid = family_ids[i]
            fname = (
                self.inverse_family_map.get(fid)
                if (self.inverse_family_map is not None and fid is not None)
                else None
            )
            results.append({
                "is_malware": int(is_malware[i]),
                "malware_score": float(scores[i]),
                "family_id": fid,
                "family_name": fname,
            })
        return results


def evaluate_cascade(
    layer1: UnifiedModel,
    layer2: UnifiedModel,
    binary_dir: str,
    family_dir: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate cascade end-to-end on aligned test splits."""
    # Dùng CascadeClassifier thay vì duplicate logic threshold + routing.
    cascade = CascadeClassifier(
        layer1=layer1,
        layer2=layer2,
        threshold=threshold,
    )

    print("[cascade] load binary test", flush=True)
    x_bin, y_bin = read_vectorized_features(binary_dir, "test")
    print("[cascade] load family test", flush=True)
    x_fam, y_fam = read_vectorized_features(family_dir, "test")

    if x_bin.shape[0] != x_fam.shape[0]:
        raise ValueError("Binary and family test sets are not aligned by row count.")

    print("[cascade] layer1 predict", flush=True)
    y1_scores = np.asarray(layer1.predict_scores(x_bin))
    y1_pred = (y1_scores >= threshold).astype(np.int32)

    family_pred = np.full_like(y_fam, fill_value=-1)
    malware_idx = np.where(y1_pred == 1)[0]
    if len(malware_idx) > 0:
        print("[cascade] layer2 predict", flush=True)
        family_pred[malware_idx] = layer2.predict_labels(x_fam[malware_idx])

    true_malware_mask = y_bin == 1
    true_family_mask = y_fam != -1
    malware_recall = float(np.mean(y1_pred[true_malware_mask] == 1)) if np.any(true_malware_mask) else 0.0

    end_to_end_correct = np.sum(
        (family_pred[true_family_mask] == y_fam[true_family_mask])
        & (y1_pred[true_family_mask] == 1)
    )
    strict_family_acc = float(end_to_end_correct / np.sum(true_family_mask)) if np.sum(true_family_mask) else 0.0

    return {
        "binary_accuracy": float(np.mean(y1_pred == y_bin)),
        "binary_malware_recall": malware_recall,
        "predicted_malware_count": int(np.sum(y1_pred == 1)),
        "true_family_samples": int(np.sum(true_family_mask)),
        "end_to_end_family_accuracy": strict_family_acc,
    }
