"""
Inference layer dùng cho GUI / CLI: quét 1 file, nhiều file, hoặc cả thư mục
qua cascade layer1 (benign/malware) -> layer2 (family, hoặc "unknown").

Tái sử dụng lại các thành phần đã có trong project:
- PEFeatureExtractor (features.py): trích xuất vector đặc trưng trực tiếp từ bytes.
- UnifiedModel / load_model (modeling.py): load model layer1, layer2 đã train.
- load_label_map (labels.py): map tên family <-> id dùng khi train layer2.

Không đụng tới pipeline train/vectorize dữ liệu gốc — module này chỉ phục vụ
suy luận (inference) trên file thực thi do người dùng cung cấp.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .features import PEFeatureExtractor
from .labels import load_label_map
from .modeling import UnifiedModel, load_model

# Các đuôi file thực thi phổ biến trên Windows — dùng để lọc khi quét cả thư mục.
DEFAULT_EXECUTABLE_EXTENSIONS = (
    ".exe", ".dll", ".sys", ".scr", ".ocx", ".com", ".cpl", ".drv", ".efi", ".msi",
)

UNKNOWN_FAMILY = "unknown"


def _status(message: str) -> None:
    print(f"[inference] {message}", flush=True)


@dataclass
class ScanResult:
    """Kết quả quét 1 file/mẫu qua cascade layer1 -> layer2."""

    path: str
    sha256: str | None = None
    size: int | None = None
    is_pe: bool | None = None
    is_malware: bool | None = None
    malware_score: float | None = None
    # family = None  -> benign, không chạy layer2
    # family = "unknown" -> layer2 không đủ tin cậy hoặc không có layer2/family map
    # family = "<tên>"   -> layer2 xác định được family cụ thể
    family: str | None = None
    family_confidence: float | None = None
    top_families: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict phẳng, tiện cho hiển thị bảng / xuất CSV."""
        row = asdict(self)
        row["top_families"] = "; ".join(
            f"{name} ({score:.3f})" for name, score in self.top_families
        )
        return row


class MalwareScanner:
    """Cascade 2 lớp:
    - Layer 1: benign (0) vs malware (1).
    - Layer 2: chỉ chạy cho các mẫu bị Layer 1 gắn nhãn malware, dự đoán family.
      Nếu family không đủ tin cậy (dưới `family_confidence_threshold`), hoặc
      không có model/label-map layer2, kết quả family sẽ là "unknown".
    """

    def __init__(
        self,
        layer1: UnifiedModel,
        layer2: UnifiedModel | None = None,
        family_label_map: dict[str, int] | None = None,
        layer1_threshold: float | None = None,
        family_confidence_threshold: float = 0.05,
        top_k: int = 3,
    ) -> None:
        self.layer1 = layer1
        self.layer2 = layer2
        self.family_label_map = family_label_map or {}
        self.inverse_family_map = {v: k for k, v in self.family_label_map.items()}
        self.layer1_threshold = (
            float(layer1_threshold)
            if layer1_threshold is not None
            else float(getattr(layer1, "threshold", 0.5))
        )
        self.family_confidence_threshold = float(family_confidence_threshold)
        self.top_k = max(1, int(top_k))
        self.extractor = PEFeatureExtractor()

    @classmethod
    def from_paths(
        cls,
        layer1_path: str | Path,
        layer2_path: str | Path | None = None,
        family_label_map_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "MalwareScanner":
        """Tiện ích load thẳng từ đường dẫn file — dùng cho CLI/script."""
        layer1 = load_model(layer1_path)
        layer2 = load_model(layer2_path) if layer2_path else None
        family_map = None
        if family_label_map_path and Path(family_label_map_path).exists():
            family_map = load_label_map(family_label_map_path)
        return cls(layer1=layer1, layer2=layer2, family_label_map=family_map, **kwargs)

    # -- trích xuất đặc trưng --------------------------------------------------

    def _extract(self, data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
        raw_obj = self.extractor.raw_features(data)
        vector = self.extractor.process_raw_features(raw_obj).astype(np.float32)
        general = raw_obj.get("general", {}) or {}
        meta = {
            "sha256": raw_obj.get("sha256"),
            "size": int(general.get("size", len(data))),
            "is_pe": bool(general.get("is_pe", 0)),
        }
        return vector, meta

    # -- suy ra family từ vector score của layer2 ------------------------------

    def _resolve_family(
        self, score_row: np.ndarray, classes: np.ndarray | None
    ) -> tuple[str, float, list[tuple[str, float]]]:
        score_row = np.asarray(score_row).ravel()
        n = len(score_row)
        if n == 0:
            return UNKNOWN_FAMILY, 0.0, []

        k = min(self.top_k, n)
        order = np.argsort(score_row)[::-1][:k]
        top: list[tuple[str, float]] = []
        for idx in order:
            class_id = int(classes[idx]) if classes is not None and len(classes) == n else int(idx)
            name = self.inverse_family_map.get(class_id, f"class_{class_id}")
            top.append((name, float(score_row[idx])))

        best_name, best_conf = top[0]
        # Không có label map (không biết tên family) hoặc độ tin cậy quá thấp
        # -> không xác định được family, trả về "unknown" thay vì đoán bừa.
        if not self.inverse_family_map or best_conf < self.family_confidence_threshold:
            return UNKNOWN_FAMILY, best_conf, top
        return best_name, best_conf, top

    # -- quét theo batch (vectorized, không lặp predict từng file) -------------

    def scan_many(self, items: Iterable[tuple[str, bytes]]) -> list[ScanResult]:
        names: list[str] = []
        vectors: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        results: list[ScanResult] = []

        for name, data in items:
            try:
                vector, meta = self._extract(data)
            except Exception as exc:  # file hỏng/không đọc được -> ghi lỗi, không crash cả batch
                results.append(ScanResult(path=name, error=str(exc)))
                continue
            names.append(name)
            vectors.append(vector)
            metas.append(meta)

        if not vectors:
            return results

        x = np.vstack(vectors)

        # Layer 1 cho toàn bộ batch một lần.
        scores1 = np.asarray(self.layer1.predict_scores(x), dtype=np.float64).ravel()
        is_malware = scores1 >= self.layer1_threshold

        # Layer 2 chỉ cho các dòng bị nghi là malware.
        family_scores = None
        classes = None
        malware_idx = np.where(is_malware)[0]
        if self.layer2 is not None and len(malware_idx) > 0:
            raw_scores = np.asarray(self.layer2.predict_scores(x[malware_idx]), dtype=np.float64)
            family_scores = raw_scores.reshape(len(malware_idx), -1)
            classes = self.layer2.score_classes()

        fam_pos = 0
        for i, (name, meta) in enumerate(zip(names, metas)):
            result = ScanResult(
                path=name,
                sha256=meta["sha256"],
                size=meta["size"],
                is_pe=meta["is_pe"],
                is_malware=bool(is_malware[i]),
                malware_score=float(scores1[i]),
            )
            if is_malware[i]:
                if family_scores is not None:
                    family, confidence, top = self._resolve_family(family_scores[fam_pos], classes)
                    fam_pos += 1
                    result.family = family
                    result.family_confidence = confidence
                    result.top_families = top
                else:
                    # Không có layer2/family map để tra cứu -> không xác định được family.
                    result.family = UNKNOWN_FAMILY
            results.append(result)
        return results

    # -- API tiện dụng cho 1 file ------------------------------------------------

    def scan_bytes(self, data: bytes, name: str = "buffer") -> ScanResult:
        return self.scan_many([(name, data)])[0]

    def scan_file(self, path: str | Path) -> ScanResult:
        path = Path(path)
        try:
            data = path.read_bytes()
        except Exception as exc:
            return ScanResult(path=str(path), error=str(exc))
        return self.scan_bytes(data, name=str(path))

    # -- quét cả thư mục --------------------------------------------------------

    def scan_folder(
        self,
        folder: str | Path,
        recursive: bool = True,
        extensions: Iterable[str] | None = DEFAULT_EXECUTABLE_EXTENSIONS,
        max_file_size_mb: float | None = 200.0,
        batch_size: int = 16,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[ScanResult]:
        """Quét toàn bộ file trong `folder`.

        extensions=None hoặc rỗng -> không lọc đuôi file, quét tất cả.
        max_file_size_mb=None -> không giới hạn dung lượng file.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Không tìm thấy thư mục: {folder}")

        ext_set = {e.lower() for e in extensions} if extensions else None
        max_bytes = max_file_size_mb * 1024 * 1024 if max_file_size_mb else None

        candidates: list[Path] = []
        if recursive:
            for root, _dirs, files in os.walk(folder):
                for file_name in files:
                    if ext_set is not None and Path(file_name).suffix.lower() not in ext_set:
                        continue
                    candidates.append(Path(root) / file_name)
        else:
            for file_name in os.listdir(folder):
                file_path = folder / file_name
                if not file_path.is_file():
                    continue
                if ext_set is not None and file_path.suffix.lower() not in ext_set:
                    continue
                candidates.append(file_path)

        results: list[ScanResult] = []
        total = len(candidates)
        batch: list[tuple[str, bytes]] = []

        def _flush() -> None:
            if batch:
                results.extend(self.scan_many(batch))
                batch.clear()

        for done, file_path in enumerate(candidates, start=1):
            try:
                size = file_path.stat().st_size
                if max_bytes is not None and size > max_bytes:
                    results.append(
                        ScanResult(
                            path=str(file_path),
                            size=size,
                            error=f"Bỏ qua: file lớn hơn giới hạn {max_file_size_mb} MB",
                        )
                    )
                else:
                    data = file_path.read_bytes()
                    batch.append((str(file_path), data))
            except Exception as exc:
                results.append(ScanResult(path=str(file_path), error=str(exc)))

            if len(batch) >= batch_size:
                _flush()
            if progress_callback:
                progress_callback(done, total, str(file_path))

        _flush()
        _status(f"scan_folder done: {len(results)} kết quả / {total} file ứng viên")
        return results
