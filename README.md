# EMBER2024 Malware Classification Project

This repository contains a cleaned and optimized malware-classification pipeline built around the EMBER2024 raw JSONL features.

## Final project layout

```text
EMBER2024/
├─ data_pe/
├─ project/
│  └─ malware_pipeline/
│     └─ scripts/
│        ├─ prepare_data.py
│        ├─ train_layer1.py
│        ├─ train_layer2.py
│        ├─ eval_cascade.py
│        └─ benchmark_models.py
└─ src/
   └─ thrember/
      ├─ dataset.py
      ├─ labels.py
      ├─ vectorize.py
      ├─ modeling.py
      ├─ cascade.py
      ├─ download.py
      └─ features.py
```

## What this final pipeline does

- **Layer 1**: binary classification (`label`) for benign vs malware.
- **Layer 2**: multiclass classification (`family`) for malware family prediction.
- **Cascade evaluation**: run layer 1 first, then route predicted-malware samples to layer 2.
- **Storage optimization**: family vectors reuse the binary `X_*.dat` files through `X_*.path` references, so the project does **not** duplicate the feature matrix.

## Installation

From the project root:

```bash
pip install -e .
```

## Downloading data

```python
import thrember

thrember.download_dataset('/path/to/download/to/', file_type='PE')
thrember.download_dataset('/path/to/download/to/', split='challenge')
```

Valid values:
- `split`: `all`, `train`, `test`, `challenge`
- `file_type`: `all`, `PE`, `Win32`, `Win64`, `Dot_Net`, `APK`, `ELF`, `PDF`

## Preparing project data

```bash
python project/malware_pipeline/scripts/prepare_data.py \
  --source-data-dir /path/to/data_pe \
  --binary-out-dir /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --family-out-dir /path/to/project/malware_pipeline/outputs/vectorized_family \
  --family-class-min 10 \
  --workers 1
```

Output behavior:
- `binary_out_dir` stores the real `X_*.dat` and `y_*.dat` files for layer 1.
- `family_out_dir` stores only `y_*.dat`, `family_label_map.json`, and `X_*.path` references to the binary features.

## Training layer 1

```bash
python project/malware_pipeline/scripts/train_layer1.py \
  --model lgbm \
  --data-dir /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --models-dir /path/to/project/malware_pipeline/outputs/models \
  --metrics-dir /path/to/project/malware_pipeline/outputs/metrics
```

## Training layer 2

```bash
python project/malware_pipeline/scripts/train_layer2.py \
  --model lgbm \
  --data-dir /path/to/project/malware_pipeline/outputs/vectorized_family \
  --models-dir /path/to/project/malware_pipeline/outputs/models \
  --metrics-dir /path/to/project/malware_pipeline/outputs/metrics
```

## Evaluating the cascade

```bash
python project/malware_pipeline/scripts/eval_cascade.py \
  --binary-dir /path/to/project/malware_pipeline/outputs/vectorized_binary \
  --family-dir /path/to/project/malware_pipeline/outputs/vectorized_family \
  --layer1-model /path/to/project/malware_pipeline/outputs/models/layer1_lgbm.txt \
  --layer2-model /path/to/project/malware_pipeline/outputs/models/layer2_lgbm.txt \
  --out-json /path/to/project/malware_pipeline/outputs/metrics/cascade_eval.json
```

## Notes

- `read_metadata()` now uses a low-memory path: streaming Polars when available, otherwise chunked pandas.
- LightGBM models are saved as `.txt` or `.model`; sklearn models are saved as `.joblib`.
- `load_model()` supports `.txt`, `.model`, `.joblib`, and `.pkl` payloads saved through the project helpers.

## References

- EMBER2024 dataset: <https://huggingface.co/datasets/joyce8/EMBER2024>
- ClarAVy: <https://github.com/FutureComputing4AI/ClarAVy/>


## Recommended run mode for 8GB RAM laptops

- Use `--workers 1` for `prepare_data.py` on Windows laptops similar to i5-10210U / 8GB RAM.
- Keep model benchmarking, but train one model at a time.
- Prefer LightGBM as the final cascade choice after benchmarking.
