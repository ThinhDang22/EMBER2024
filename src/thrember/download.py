from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


VALID_SPLITS = ['all', 'train', 'test', 'challenge']
VALID_FILES = ['all', 'PE', 'Win32', 'Win64', 'Dot_Net', 'APK', 'ELF', 'PDF']


def _ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Invalid directory: {path}')
    return path


def download_dataset(download_dir: Path | str, split: str = 'all', file_type: str = 'all') -> None:
    download_dir = _ensure_dir(download_dir)
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be in {', '.join(VALID_SPLITS)}")
    if file_type not in VALID_FILES:
        raise ValueError(f"file_type must be in {', '.join(VALID_FILES)}")

    splits = VALID_SPLITS[1:] if split == 'all' else [split]

    if file_type == 'PE':
        file_types = ['Win32', 'Win64', 'Dot_Net']
    elif file_type == 'all':
        file_types = VALID_FILES[2:]
    else:
        file_types = [file_type]

    for current_split in splits:
        if current_split == 'challenge':
            zip_path = hf_hub_download(repo_id='joyce8/EMBER2024', filename='challenge.zip', repo_type='dataset')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(download_dir)
            continue

        for current_type in file_types:
            file_name = f'{current_type}_{current_split}.zip'
            zip_path = hf_hub_download(repo_id='joyce8/EMBER2024', filename=file_name, repo_type='dataset')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(download_dir)


def download_models(download_dir: Path | str) -> None:
    download_dir = _ensure_dir(download_dir)
    repo_id = 'joyce8/EMBER2024-benchmark-models'
    model_files = [
        file_name
        for file_name in list_repo_files(repo_id)
        if file_name.endswith(('.model', '.txt', '.joblib', '.json'))
    ]
    for file_name in model_files:
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=str(download_dir))
