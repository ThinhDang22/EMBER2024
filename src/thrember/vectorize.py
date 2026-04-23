from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataset import count_jsonl_rows, gather_feature_paths, raw_feature_iterator
from .features import PEFeatureExtractor
from .labels import build_label_map, save_label_map


_WORKER_STATE: dict[str, object] = {}


def _write_y_shape_meta(y_path: Path, shape: tuple[int, ...]) -> None:
    meta_path = y_path.with_suffix('.shape.json')
    with meta_path.open('w', encoding='utf-8') as fout:
        json.dump({'shape': list(shape)}, fout, indent=2, ensure_ascii=False)


def _print_stage(message: str) -> None:
    print(f'[vectorize] {message}', flush=True)


def _progress_bar(desc: str, total: int):
    # mininterval keeps terminal updates lightweight so the progress bar stays useful
    # without noticeably slowing down vectorization.
    return tqdm(
        total=total,
        desc=desc,
        unit='rows',
        mininterval=1.0,
        dynamic_ncols=True,
        leave=True,
    )


def _init_vectorize_worker(
    x_path: str,
    y_path: str,
    nrows: int,
    feature_dim: int,
    label_type: str,
    label_map: dict[str, int],
) -> None:
    extractor = PEFeatureExtractor()
    x_mem = np.memmap(x_path, dtype=np.float32, mode='r+', shape=(nrows, feature_dim))

    if label_type in {'label', 'family'}:
        y_shape = (nrows,)
    else:
        y_shape = (nrows, len(label_map))
    y_mem = np.memmap(y_path, dtype=np.int32, mode='r+', shape=y_shape)

    _WORKER_STATE.clear()
    _WORKER_STATE.update({
        'extractor': extractor,
        'x_mem': x_mem,
        'y_mem': y_mem,
        'label_type': label_type,
        'label_map': label_map,
    })


def _vectorize_row(task: tuple[int, str]) -> int:
    irow, raw_features_string = task
    raw_features = json.loads(raw_features_string)
    extractor: PEFeatureExtractor = _WORKER_STATE['extractor']  # type: ignore[assignment]
    x_mem: np.memmap = _WORKER_STATE['x_mem']  # type: ignore[assignment]
    y_mem: np.memmap = _WORKER_STATE['y_mem']  # type: ignore[assignment]
    label_type: str = _WORKER_STATE['label_type']  # type: ignore[assignment]
    label_map: dict[str, int] = _WORKER_STATE['label_map']  # type: ignore[assignment]

    x_mem[irow] = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError(f'Invalid label_type: {label_type}')

    label = raw_features[label_type]
    if label_type in {'label', 'family'}:
        if label is None:
            y_mem[irow] = -1
        elif isinstance(label, int):
            y_mem[irow] = label
        elif isinstance(label, str):
            y_mem[irow] = label_map.get(label, -1)
        else:
            raise ValueError(f'Unable to parse label format for {label_type}: {type(label)}')
    else:
        if label is not None:
            if not isinstance(label, list):
                raise ValueError(f'Expected list label for multilabel task {label_type}')
            for single_label in label:
                mapped = label_map.get(single_label)
                if mapped is not None:
                    y_mem[irow, mapped] = 1

    return 1


def _iter_tasks(raw_feature_paths: list[Path]):
    for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)):
        yield (irow, raw_features_string)


def _vectorize_subset_single_worker(
    x_path: Path,
    y_path: Path,
    raw_feature_paths: list[Path],
    nrows: int,
    feature_dim: int,
    label_type: str,
    label_map: dict[str, int],
    subset_name: str,
) -> None:
    extractor = PEFeatureExtractor()
    x_mem = np.memmap(x_path, dtype=np.float32, mode='r+', shape=(nrows, feature_dim))
    if label_type in {'label', 'family'}:
        y_shape = (nrows,)
    else:
        y_shape = (nrows, len(label_map))
    y_mem = np.memmap(y_path, dtype=np.int32, mode='r+', shape=y_shape)

    with _progress_bar(f'Vectorizing {subset_name} subset', nrows) as pbar:
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths), start=1):
            raw_features = json.loads(raw_features_string)
            x_mem[irow - 1] = extractor.process_raw_features(raw_features)

            if label_type not in raw_features:
                raise ValueError(f'Invalid label_type: {label_type}')
            label = raw_features[label_type]

            if label_type in {'label', 'family'}:
                if label is None:
                    y_mem[irow - 1] = -1
                elif isinstance(label, int):
                    y_mem[irow - 1] = label
                elif isinstance(label, str):
                    y_mem[irow - 1] = label_map.get(label, -1)
                else:
                    raise ValueError(f'Unable to parse label format for {label_type}: {type(label)}')
            else:
                if label is not None:
                    if not isinstance(label, list):
                        raise ValueError(f'Expected list label for multilabel task {label_type}')
                    for single_label in label:
                        mapped = label_map.get(single_label)
                        if mapped is not None:
                            y_mem[irow - 1, mapped] = 1

            pbar.update(1)

    x_mem.flush()
    y_mem.flush()


def vectorize_subset(
    x_path: Path | str,
    y_path: Path | str,
    raw_feature_paths: list[Path],
    nrows: int,
    label_type: str = 'label',
    label_map: dict[str, int] | None = None,
    workers: int | None = None,
    subset_name: str | None = None,
) -> None:
    """Vectorize one subset and write X/y to disk."""
    label_map = label_map or {}
    x_path = Path(x_path)
    y_path = Path(y_path)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    y_path.parent.mkdir(parents=True, exist_ok=True)
    subset_name = subset_name or x_path.stem.replace('X_', '')

    feature_dim = PEFeatureExtractor().dim

    x = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(nrows, feature_dim))
    x.flush()
    del x

    if label_type in {'label', 'family'}:
        y_shape = (nrows,)
        y = np.memmap(y_path, dtype=np.int32, mode='w+', shape=y_shape)
        y[:] = -1
    else:
        y_shape = (nrows, len(label_map))
        y = np.memmap(y_path, dtype=np.int32, mode='w+', shape=y_shape)
        y[:] = 0
    y.flush()
    del y
    _write_y_shape_meta(y_path, y_shape)

    if nrows == 0:
        _print_stage(f'{subset_name} skipped (0 rows)')
        return

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers or max_workers, max_workers)
    _print_stage(f'{subset_name} start | rows={nrows} | workers={workers}')
    start = time.perf_counter()

    if workers <= 1:
        _vectorize_subset_single_worker(
            x_path,
            y_path,
            raw_feature_paths,
            nrows,
            feature_dim,
            label_type,
            label_map,
            subset_name,
        )
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(
            processes=workers,
            initializer=_init_vectorize_worker,
            initargs=(str(x_path), str(y_path), nrows, feature_dim, label_type, label_map),
        ) as pool:
            with _progress_bar(f'Vectorizing {subset_name} subset', nrows) as pbar:
                for _ in pool.imap_unordered(_vectorize_row, _iter_tasks(raw_feature_paths), chunksize=128):
                    pbar.update(1)

    elapsed = time.perf_counter() - start
    _print_stage(f'{subset_name} done | {elapsed:.2f}s')


def create_vectorized_features(
    data_dir: Path | str,
    label_type: str = 'label',
    class_min: int = 10,
    output_dir: Path | str | None = None,
    save_label_map_file: bool = True,
    workers: int | None = None,
) -> dict[str, int]:
    """Create vectorized train/test/challenge files for one task."""
    data_path = Path(data_dir)
    out_path = Path(output_dir) if output_dir is not None else data_path
    out_path.mkdir(parents=True, exist_ok=True)

    subsets = {
        'train': gather_feature_paths(data_path, 'train'),
        'test': gather_feature_paths(data_path, 'test'),
        'challenge': gather_feature_paths(data_path, 'challenge'),
    }
    nrows = {subset: count_jsonl_rows(paths) for subset, paths in subsets.items()}

    _print_stage(f'build label map: {label_type}')
    label_map = build_label_map(data_path, label_type, class_min=class_min)

    for subset, paths in subsets.items():
        vectorize_subset(
            out_path / f'X_{subset}.dat',
            out_path / f'y_{subset}.dat',
            paths,
            nrows[subset],
            label_type=label_type,
            label_map=label_map,
            workers=workers,
            subset_name=subset,
        )

    if save_label_map_file and label_type != 'label':
        save_label_map(out_path / f'{label_type}_label_map.json', label_map)

    return label_map


def _write_x_reference(reference_file: Path, target_file: Path) -> None:
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target_file.resolve(), start=reference_file.parent.resolve())
    reference_file.write_text(relative, encoding='utf-8')


def _create_label_only_subset(
    y_path: Path,
    raw_feature_paths: list[Path],
    nrows: int,
    label_type: str,
    label_map: dict[str, int],
    subset_name: str,
) -> None:
    _print_stage(f'{subset_name} family labels start | rows={nrows}')
    start = time.perf_counter()

    if label_type in {'label', 'family'}:
        y = np.memmap(y_path, dtype=np.int32, mode='w+', shape=(nrows,))
        y[:] = -1
        with _progress_bar(f'Building labels {subset_name} subset', nrows) as pbar:
            for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths), start=1):
                raw_features = json.loads(raw_features_string)
                label = raw_features.get(label_type)
                if label is None:
                    y[irow - 1] = -1
                elif isinstance(label, int):
                    y[irow - 1] = label
                elif isinstance(label, str):
                    y[irow - 1] = label_map.get(label, -1)
                else:
                    raise ValueError(f'Unable to parse label format for {label_type}: {type(label)}')
                pbar.update(1)
        y.flush()
        del y
        _write_y_shape_meta(y_path, (nrows,))
    else:
        y = np.memmap(y_path, dtype=np.int32, mode='w+', shape=(nrows, len(label_map)))
        y[:] = 0
        with _progress_bar(f'Building labels {subset_name} subset', nrows) as pbar:
            for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths), start=1):
                raw_features = json.loads(raw_features_string)
                labels = raw_features.get(label_type)
                if labels is not None:
                    if not isinstance(labels, list):
                        raise ValueError(f'Expected list label for multilabel task {label_type}')
                    for single_label in labels:
                        mapped = label_map.get(single_label)
                        if mapped is not None:
                            y[irow - 1, mapped] = 1
                pbar.update(1)
        y.flush()
        del y
        _write_y_shape_meta(y_path, (nrows, len(label_map)))

    elapsed = time.perf_counter() - start
    _print_stage(f'{subset_name} family labels done | {elapsed:.2f}s')


def create_project_vectorized_features(
    source_data_dir: Path | str,
    binary_out_dir: Path | str,
    family_out_dir: Path | str,
    family_class_min: int = 10,
    workers: int | None = None,
) -> dict[str, dict[str, int]]:
    """Create the binary/family vectorized layout used by the cascade pipeline."""
    source_data_dir = Path(source_data_dir)
    binary_out_dir = Path(binary_out_dir)
    family_out_dir = Path(family_out_dir)
    binary_out_dir.mkdir(parents=True, exist_ok=True)
    family_out_dir.mkdir(parents=True, exist_ok=True)

    _print_stage('binary dataset start')
    create_vectorized_features(
        data_dir=source_data_dir,
        label_type='label',
        output_dir=binary_out_dir,
        save_label_map_file=False,
        workers=workers,
    )
    _print_stage('binary dataset done')

    _print_stage('family label map start')
    family_label_map = build_label_map(source_data_dir, 'family', class_min=family_class_min)
    save_label_map(family_out_dir / 'family_label_map.json', family_label_map)
    _print_stage(f'family label map done | classes={len(family_label_map)}')

    for subset in ('train', 'test', 'challenge'):
        raw_paths = gather_feature_paths(source_data_dir, subset)
        nrows = count_jsonl_rows(raw_paths)
        _create_label_only_subset(
            family_out_dir / f'y_{subset}.dat',
            raw_paths,
            nrows,
            label_type='family',
            label_map=family_label_map,
            subset_name=subset,
        )
        _write_x_reference(
            family_out_dir / f'X_{subset}.path',
            binary_out_dir / f'X_{subset}.dat',
        )

    _print_stage('project vectorization done')
    return {'binary': {}, 'family': family_label_map}
