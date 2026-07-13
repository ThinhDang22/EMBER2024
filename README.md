# EMBER2024 Malware Classification Project

Pipeline phân loại malware 2 tầng (cascade) xây dựng trên bộ đặc trưng tĩnh
**EMBER2024**, kèm theo **giao diện quét file** (Streamlit) và **CLI** để
xác thực một file/tập file thực thi bất kỳ ngoài đời thực.

- **Layer 1** — phân loại **benign / malware** (nhị phân).
- **Layer 2** — nếu Layer 1 báo malware, phân loại tiếp **family** (đa lớp,
  3046 family trong bộ EMBER2024). Nếu không đủ tin cậy để xác định family,
  kết quả sẽ là **`unknown`** thay vì đoán bừa.
- **Giao diện quét (GUI/CLI)** — kéo-thả 1 file, nhiều file, hoặc trỏ tới cả
  một thư mục chứa file thực thi để quét hàng loạt qua cascade layer1 → layer2.

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [1. Tải dữ liệu](#1-tải-dữ-liệu)
- [2. Chuẩn bị dữ liệu huấn luyện](#2-chuẩn-bị-dữ-liệu-huấn-luyện)
- [3. Huấn luyện Layer 1 (benign/malware)](#3-huấn-luyện-layer-1-benignmalware)
- [4. Huấn luyện Layer 2 (family)](#4-huấn-luyện-layer-2-family)
- [5. Đánh giá cascade](#5-đánh-giá-cascade)
- [6. So sánh model (benchmark)](#6-so-sánh-model-benchmark)
- [7. Giao diện quét file (GUI — Streamlit)](#7-giao-diện-quét-file-gui--streamlit)
- [8. Quét bằng dòng lệnh (CLI)](#8-quét-bằng-dòng-lệnh-cli)
- [9. Dùng trực tiếp trong Python](#9-dùng-trực-tiếp-trong-python)
- [Logic xác định "unknown"](#logic-xác-định-unknown)
- [Ghi chú kỹ thuật](#ghi-chú-kỹ-thuật)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Cấu trúc thư mục

```text
EMBER2024/
├─ README.md
├─ pyproject.toml
├─ setup.cfg
├─ src/
│  └─ thrember/
│     ├─ features.py        # PEFeatureExtractor — trích xuất đặc trưng tĩnh từ PE
│     ├─ vectorize.py        # JSONL -> ma trận đặc trưng .dat (vectorize hàng loạt)
│     ├─ dataset.py          # đọc metadata / ma trận đặc trưng đã vectorize
│     ├─ labels.py           # build/load/save family_label_map.json
│     ├─ modeling.py         # UnifiedModel: train/save/load/predict (lgbm, catboost, sklearn...)
│     ├─ cascade.py          # CascadeClassifier — ghép layer1 + layer2, evaluate_cascade
│     ├─ inference.py        # MalwareScanner — quét file/thư mục thực tế, logic "unknown"
│     └─ download.py         # tải dataset & model có sẵn từ Hugging Face
└─ project/
   └─ malware_pipeline/
      ├─ scripts/
      │  ├─ prepare_data.py          # JSONL -> vectorized_binary/ + vectorized_family/
      │  ├─ train_layer1.py          # train model layer 1
      │  ├─ train_layer2.py          # train model layer 2 (lgbm/lgbm_fast/lgbm_ova/catboost/pa)
      │  ├─ train_catboost_layer2.py # train CatBoost layer 2 quy mô lớn (máy nhiều RAM)
      │  ├─ eval_cascade.py          # đánh giá end-to-end layer1 -> layer2
      │  ├─ benchmark_models.py      # so sánh nhiều loại model
      │  └─ scan_cli.py              # quét file/thư mục qua dòng lệnh (không cần GUI)
      ├─ app/
      │  └─ app.py                  # giao diện Streamlit: quét 1 file / nhiều file / cả thư mục
      └─ outputs/                    # models/, metrics/, vectorized_binary/, vectorized_family/, logs/
```

> Thư mục `outputs/` không có sẵn trong repo — sẽ được tạo ra khi bạn chạy các
> script `prepare_data.py`, `train_layer1.py`, `train_layer2.py`, ... ở bước dưới.

## Cài đặt

Yêu cầu Python >= 3.10.

```bash
# Cài package + toàn bộ dependency train/vectorize (lightgbm, catboost, pefile, polars...)
pip install -e .

# Cài thêm để chạy giao diện Streamlit (streamlit, pandas)
pip install -e ".[app]"
```

## 1. Tải dữ liệu

```python
import thrember

thrember.download_dataset('/path/to/download/to/', file_type='PE')
thrember.download_dataset('/path/to/download/to/', split='challenge')

# Tải luôn model đã pre-train sẵn (nếu có) từ Hugging Face
thrember.download_models('/path/to/models/')
```

Giá trị hợp lệ:
- `split`: `all`, `train`, `test`, `challenge`
- `file_type`: `all`, `PE`, `Win32`, `Win64`, `Dot_Net`, `APK`, `ELF`, `PDF`

## 2. Chuẩn bị dữ liệu huấn luyện

```bash
python project/malware_pipeline/scripts/prepare_data.py \
  --source-data-dir /path/to/data_pe \
  --binary-out-dir  /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --family-out-dir  /path/to/project/malware_pipeline/outputs/vectorized_family \
  --family-class-min 10 \
  --workers 1
```

- `--binary-out-dir`: chứa `X_*.dat` + `y_*.dat` thật, dùng để train **Layer 1**.
- `--family-out-dir`: chỉ chứa `y_*.dat`, `family_label_map.json`, và `X_*.path`
  (con trỏ trỏ ngược về file đặc trưng ở `binary_out_dir`) — **không** nhân bản
  ma trận đặc trưng, tiết kiệm dung lượng đĩa.
- `--family-class-min`: số mẫu tối thiểu để 1 family được coi là 1 lớp riêng
  (family ít mẫu hơn sẽ không lọt vào tập train layer 2).

## 3. Huấn luyện Layer 1 (benign/malware)

```bash
python project/malware_pipeline/scripts/train_layer1.py \
  --model lgbm \
  --data-dir    /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --models-dir  /path/to/project/malware_pipeline/outputs/models \
  --metrics-dir /path/to/project/malware_pipeline/outputs/metrics
```

Tuỳ chọn thêm: `--model` (`lgbm`/`logreg`/`rf`/`mlp`), `--threshold` (ép ngưỡng
malware cụ thể thay vì để script tự chọn ngưỡng tối ưu), `--sample-size`
(train nhanh trên tập con để thử nghiệm), `--logs-dir`.

## 4. Huấn luyện Layer 2 (family)

```bash
python project/malware_pipeline/scripts/train_layer2.py \
  --model lgbm \
  --data-dir    /path/to/project/malware_pipeline/outputs/vectorized_family \
  --models-dir  /path/to/project/malware_pipeline/outputs/models \
  --metrics-dir /path/to/project/malware_pipeline/outputs/metrics
```

`--model` hỗ trợ: `lgbm`, `lgbm_fast`, `lgbm_ova`, `catboost`, `pa` (Passive-Aggressive,
baseline dạng streaming). Thêm `--validation-size`, `--random-state`,
`--sample-size` để tinh chỉnh.

> Với máy có rất nhiều RAM (server), có thể dùng
> `project/malware_pipeline/scripts/train_catboost_layer2.py` — bản CatBoost
> tối ưu bộ nhớ/disk cho quy mô 3046 family (xem docstring đầu file để biết
> các flag `--budget-usd`, `--chunk-size`, `--preset`, `--no-snapshot`...).

## 5. Đánh giá cascade

```bash
python project/malware_pipeline/scripts/eval_cascade.py \
  --binary-dir  /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --family-dir  /path/to/project/malware_pipeline/outputs/vectorized_family \
  --layer1-model /path/to/project/malware_pipeline/outputs/models/layer1_lgbm.txt \
  --layer2-model /path/to/project/malware_pipeline/outputs/models/layer2_lgbm.txt \
  --out-json    /path/to/project/malware_pipeline/outputs/metrics/cascade_eval.json
```

Script này chạy toàn bộ tập test qua `CascadeClassifier` (Layer 1 → Layer 2)
và xuất báo cáo end-to-end (accuracy layer 1, accuracy family trên các mẫu
đúng là malware, v.v.) ra file JSON.

## 6. So sánh model (benchmark)

```bash
python project/malware_pipeline/scripts/benchmark_models.py \
  --binary-dir /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --family-dir /path/to/project/malware_pipeline/outputs/vectorized_family \
  --out-json   /path/to/project/malware_pipeline/outputs/metrics/benchmark.json
```

## 7. Giao diện quét file (GUI — Streamlit)

Sau khi đã có model Layer 1 (bắt buộc) và Layer 2 (khuyến nghị, để biết family),
chạy:

```bash
streamlit run project/malware_pipeline/app/app.py
```

Trình duyệt sẽ mở `http://localhost:8501`. Giao diện gồm:

- **Sidebar — Cấu hình model**: đường dẫn model Layer 1, Layer 2,
  `family_label_map.json`; ngưỡng malware (Layer 1); ngưỡng tin cậy family
  (Layer 2, dùng để quyết định khi nào trả về `unknown`); số family gợi ý
  hiển thị (top-k).
- **Tab "📄 Quét file"**: upload 1 file hoặc nhiều file thực thi cùng lúc
  (chọn nhiều file trong hộp thoại = quét cả "một tập file trong 1 folder").
- **Tab "📁 Quét thư mục"**: nhập trực tiếp đường dẫn 1 thư mục trên máy,
  có tuỳ chọn quét đệ quy thư mục con, lọc theo đuôi file thực thi
  (`.exe .dll .sys .scr .ocx .com .cpl .drv .efi .msi`) hoặc quét tất cả,
  giới hạn dung lượng file tối đa (bỏ qua file quá lớn), có thanh tiến trình.
- **Tab "ℹ️ Hướng dẫn"**: giải thích cách hoạt động và lưu ý sử dụng.

Kết quả hiển thị dạng bảng gồm: kết quả (Malware/Benign), điểm malware,
family, độ tin cậy family, top family gợi ý, SHA256, kích thước, có phải PE
hợp lệ hay không, lỗi (nếu có) — kèm nút tải kết quả ra CSV.

## 8. Quét bằng dòng lệnh (CLI)

Dùng khi không cần mở giao diện (ví dụ chạy trên server / tích hợp vào script khác):

```bash
# Quét 1 file
python project/malware_pipeline/scripts/scan_cli.py \
  --target /path/to/sample.exe \
  --layer1-model outputs/models/layer1_lgbm.txt \
  --layer2-model outputs/models/layer2_lgbm.txt \
  --family-map outputs/vectorized_family/family_label_map.json

# Quét cả thư mục (đệ quy), xuất kết quả ra CSV
python project/malware_pipeline/scripts/scan_cli.py \
  --target /path/to/samples_folder \
  --layer1-model outputs/models/layer1_lgbm.txt \
  --layer2-model outputs/models/layer2_lgbm.txt \
  --family-map outputs/vectorized_family/family_label_map.json \
  --out-csv outputs/metrics/scan_result.csv
```

Các cờ hữu ích: `--no-recursive` (không quét thư mục con), `--all-files` (bỏ
lọc đuôi file), `--max-file-size-mb`, `--family-confidence-threshold`, `--top-k`.

## 9. Dùng trực tiếp trong Python

Toàn bộ logic quét (GUI và CLI đều dùng chung) nằm ở `thrember.MalwareScanner`:

```python
import thrember

scanner = thrember.MalwareScanner.from_paths(
    layer1_path="outputs/models/layer1_lgbm.txt",
    layer2_path="outputs/models/layer2_lgbm.txt",
    family_label_map_path="outputs/vectorized_family/family_label_map.json",
    family_confidence_threshold=0.05,  # dưới ngưỡng này -> "unknown"
    top_k=3,
)

# 1 file
result = scanner.scan_file("sample.exe")
print(result.is_malware, result.malware_score, result.family, result.family_confidence)

# nhiều file/bytes cùng lúc (vectorized, nhanh hơn gọi lẻ từng file)
results = scanner.scan_many([("a.exe", open("a.exe", "rb").read()),
                              ("b.exe", open("b.exe", "rb").read())])

# cả thư mục
results = scanner.scan_folder("samples_folder", recursive=True)
for r in results:
    print(r.path, r.is_malware, r.family)
```

`ScanResult` là dataclass gồm: `path`, `sha256`, `size`, `is_pe`, `is_malware`,
`malware_score`, `family`, `family_confidence`, `top_families`, `error`.
Gọi `result.to_dict()` để lấy dict phẳng (tiện ghi CSV/JSON).

## Logic xác định "unknown"

1. Nếu **Layer 1** cho điểm malware **dưới ngưỡng** → kết luận **benign**,
   không chạy Layer 2, `family = None`.
2. Nếu Layer 1 kết luận **malware** nhưng **chưa nạp model Layer 2** (hoặc
   chưa có `family_label_map.json`) → `family = "unknown"`.
3. Nếu Layer 2 chạy được nhưng xác suất family cao nhất **thấp hơn ngưỡng
   tin cậy** (mặc định 0.05, chỉnh được ở sidebar/CLI) → `family = "unknown"`
   thay vì trả về family có xác suất thấp/không đáng tin.
4. Chỉ khi Layer 2 đủ tin cậy và tra được tên family từ `family_label_map.json`
   → trả về đúng tên family (ví dụ `"emotet"`, `"redline"`, ...).

Ngưỡng tin cậy này **không cố định "đúng" tuyệt đối** — với ~3046 family và
phân bố rất lệch, hãy tự tinh chỉnh dựa trên `eval_cascade.py`/tập validation
của bạn để cân bằng giữa việc gọi tên family sai vs. gắn `unknown` quá nhiều.

## Ghi chú kỹ thuật

- `read_metadata()` dùng đường low-memory: streaming bằng Polars khi có, nếu
  không thì fallback sang pandas theo chunk.
- Model LightGBM lưu dạng `.txt`/`.model`; model sklearn lưu dạng `.joblib`.
  `load_model()` hỗ trợ cả `.txt`, `.model`, `.joblib`, `.pkl`.
- `PEFeatureExtractor` vẫn trích xuất được đặc trưng cho file **không phải PE
  hợp lệ** (không parse được header PE) — các nhóm đặc trưng phụ thuộc PE sẽ
  trả về giá trị rỗng/0, không làm chương trình crash; kết quả `is_pe=False`
  trong `ScanResult` giúp bạn biết để diễn giải kết quả thận trọng hơn.
- `MalwareScanner.scan_many()`/`scan_folder()` gọi predict theo batch
  (vectorized) thay vì lặp từng file, giúp quét nhanh hơn đáng kể với số
  lượng file lớn.
- Máy cấu hình yếu (ví dụ laptop 8GB RAM, i5-10210U): dùng `--workers 1` cho
  `prepare_data.py`, benchmark model nhưng train từng model một, ưu tiên
  LightGBM làm lựa chọn cascade cuối cùng.

## Tài liệu tham khảo

- EMBER2024 dataset: <https://huggingface.co/datasets/joyce8/EMBER2024>
- ClarAVy (gán nhãn family): <https://github.com/FutureComputing4AI/ClarAVy/>

## Cảnh báo sử dụng

Đây là công cụ hỗ trợ **nghiên cứu/phân loại** dựa trên đặc trưng tĩnh, không
phải phần mềm diệt virus đầy đủ tính năng và không nên là tuyến phòng thủ duy
nhất cho hệ thống thực tế. Giao diện Streamlit và CLI chạy hoàn toàn cục bộ —
file được quét không được gửi ra ngoài máy đang chạy chương trình.
