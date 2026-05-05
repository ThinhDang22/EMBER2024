from __future__ import annotations

import json
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import read_vectorized_features
from .features import PEFeatureExtractor


LGBM_CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


def _best_binary_threshold_by_f1(y_true, scores):
    """Select a binary classification threshold that maximizes F1 on validation data."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(y_true) == 0 or len(scores) == 0:
        return 0.5, 0.0

    candidate_thresholds = np.unique(
        np.quantile(scores, np.linspace(0.01, 0.99, 99))
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        labels = (scores >= threshold).astype(np.int32)
        f1 = f1_score(y_true, labels, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def _tpr_at_max_fpr(fpr, tpr, max_fpr: float = 0.01):
    """Return the best TPR while FPR is still <= max_fpr."""
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def _status(message: str) -> None:
    print(f'[modeling] {message}', flush=True)


def _recommended_threads() -> int:
    env_threads = os.getenv("THREMBER_NUM_THREADS")
    if env_threads:
        return max(1, int(env_threads))
    return max(1, min(4, os.cpu_count() or 1))


@dataclass
class FitConfig:
    model_type: str
    problem_type: str
    validation_size: float = 0.1
    random_state: int = 42
    sample_size: int | None = None
    params: dict[str, Any] | None = None


class UnifiedModel:
    """One wrapper class for all supported model families."""

    def __init__(
        self,
        model_type: str,
        problem_type: str,
        params: dict[str, Any] | None = None,
        categorical_features: list[int] | None = None,
    ) -> None:
        self.model_type = model_type.lower()
        self.problem_type = problem_type.lower()
        self.params = params or {}
        self.categorical_features = categorical_features or LGBM_CATEGORICAL_FEATURES
        self.model = None
        self.threshold = 0.5
        self.validation_f1 = None

    def _build_sklearn_model(self):
        if self.model_type == 'logreg':
            return make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(**self.params),
            )

        if self.model_type == 'sgd_logreg':
            return make_pipeline(
                StandardScaler(with_mean=False),
                SGDClassifier(**self.params),
            )

        if self.model_type == 'sgd_huber':
            return make_pipeline(
                StandardScaler(with_mean=False),
                SGDClassifier(**self.params),
            )

        if self.model_type == 'sgd_hinge':
            return make_pipeline(
                StandardScaler(with_mean=False),
                SGDClassifier(**self.params),
            )

        if self.model_type == 'ridge':
            return make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifier(**self.params),
            )

        if self.model_type == 'rf':
            return RandomForestClassifier(**self.params)

        if self.model_type == 'mlp':
            return make_pipeline(
                StandardScaler(with_mean=False),
                MLPClassifier(**self.params),
            )

        raise ValueError(f'Unsupported sklearn model_type: {self.model_type}')

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
    ) -> 'UnifiedModel':
        if self.model_type == 'lgbm':
            self.model = self._fit_lightgbm(
                x_train,
                y_train,
                x_val,
                y_val,
                sample_weight=sample_weight,
                val_sample_weight=val_sample_weight,
            )
        else:
            self.model = self._build_sklearn_model()
            # Most sklearn alternatives used in Layer 2 already receive class_weight
            # in params. Passing sample_weight through a Pipeline requires step-specific
            # kwargs and is intentionally avoided here to keep Layer 1 compatibility.
            self.model.fit(x_train, y_train)

        if (
            self.problem_type == 'binary'
            and x_val is not None
            and y_val is not None
            and len(y_val) > 0
        ):
            try:
                val_scores = np.asarray(self.predict_scores(x_val))
                self.threshold, self.validation_f1 = _best_binary_threshold_by_f1(y_val, val_scores)
                _status(
                    f'validation threshold selected: threshold={self.threshold:.6f}, '
                    f'f1={self.validation_f1:.6f}'
                )
            except Exception as exc:
                _status(f'could not select validation threshold: {exc}')

        return self

    def _fit_lightgbm(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
    ):
        params = dict(self.params)
        early_stopping_rounds = int(params.pop('early_stopping_rounds', 50))
        log_period = int(params.pop('log_period', 50))
        params.setdefault('num_threads', _recommended_threads())

        if self.problem_type == 'binary':
            params.setdefault('objective', 'binary')
            params.setdefault('metric', ['auc', 'binary_logloss'])
        elif self.problem_type == 'multiclass':
            params.setdefault('objective', 'multiclass')
            params.setdefault('metric', 'multi_logloss')
            params['num_class'] = int(np.max(y_train) + 1)
        else:
            raise ValueError(f'LightGBM in UnifiedModel supports binary/multiclass only, got {self.problem_type}')

        train_set = lgb.Dataset(
            x_train,
            y_train,
            weight=sample_weight,
            categorical_feature=self.categorical_features,
            free_raw_data=True,
        )

        valid_sets = None
        valid_names = None
        callbacks = []
        if x_val is not None and y_val is not None and len(y_val) > 0:
            val_set = lgb.Dataset(
                x_val,
                y_val,
                weight=val_sample_weight,
                reference=train_set,
                categorical_feature=self.categorical_features,
                free_raw_data=True,
            )
            valid_sets = [val_set]
            valid_names = ['valid']
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            if log_period > 0:
                callbacks.append(lgb.log_evaluation(period=log_period))

        return lgb.train(
            params,
            train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks or None,
        )

    def _ensure_fitted(self) -> None:
        if self.model is None:
            raise ValueError('Model is not fitted or loaded yet.')

    def score_classes(self):
        """Return the class IDs corresponding to score/probability columns, when known."""
        self._ensure_fitted()

        if self.model_type == 'lgbm':
            if self.problem_type == 'multiclass':
                try:
                    return np.arange(int(self.model.num_model_per_iteration()), dtype=np.int32)
                except Exception:
                    return None
            return None

        if isinstance(self.model, dict) and self.model.get('kind') == 'sgd_stream':
            classes = self.model.get('classes')
            return None if classes is None else np.asarray(classes, dtype=np.int32)

        estimator = self.model
        if hasattr(estimator, 'steps') and estimator.steps:
            estimator = estimator.steps[-1][1]
        if hasattr(estimator, 'classes_'):
            return np.asarray(estimator.classes_, dtype=np.int32)
        return None

    def predict_scores(self, x):
        self._ensure_fitted()

        if self.model_type == 'lgbm':
            return self.model.predict(x)

        # Streaming SGD models are stored as a small dict: scaler + classifier.
        # Transform only the current chunk to avoid copying the full dataset.
        if isinstance(self.model, dict) and self.model.get('kind') == 'sgd_stream':
            scaler = self.model['scaler']
            classifier = self.model['classifier']
            x_chunk = np.array(x, dtype=np.float32, copy=True)
            x_chunk = scaler.transform(x_chunk, copy=False)

            if hasattr(classifier, 'predict_proba'):
                probs = classifier.predict_proba(x_chunk)
                if self.problem_type == 'binary':
                    return probs[:, 1]
                return probs
            if hasattr(classifier, 'decision_function'):
                return classifier.decision_function(x_chunk)
            return classifier.predict(x_chunk)

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(x)
            if self.problem_type == 'binary':
                return probs[:, 1]
            return probs

        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(x)

        return self.model.predict(x)

    def predict_labels(self, x, threshold: float | None = None):
        scores = self.predict_scores(x)
        arr = np.asarray(scores)

        if self.problem_type == 'binary':
            if threshold is None:
                threshold = getattr(self, 'threshold', 0.5)
            return (arr >= threshold).astype(np.int32)
        if self.problem_type == 'multiclass':
            if arr.ndim == 1:
                return arr.astype(np.int32)
            idx = np.argmax(arr, axis=1).astype(np.int32)
            classes = self.score_classes()
            if classes is not None and len(classes) == arr.shape[1]:
                return classes[idx].astype(np.int32)
            return idx
        if self.problem_type == 'multilabel':
            return (arr >= threshold).astype(np.int32)

        raise ValueError(f'Unsupported problem_type: {self.problem_type}')

    def save(self, path: Path | str) -> None:
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == 'lgbm':
            self.model.save_model(str(path))
            meta_path = path.with_suffix(path.suffix + '.meta.json')
            with meta_path.open('w', encoding='utf-8') as fout:
                json.dump(
                    {
                        'model_type': self.model_type,
                        'problem_type': self.problem_type,
                        'params': self.params,
                        'threshold': self.threshold,
                        'validation_f1': self.validation_f1,
                    },
                    fout,
                    indent=2,
                    ensure_ascii=False,
                )
            return

        joblib.dump(
            {
                'model_type': self.model_type,
                'problem_type': self.problem_type,
                'params': self.params,
                'model': self.model,
                'threshold': self.threshold,
                'validation_f1': self.validation_f1,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, model_type: str | None = None, problem_type: str | None = None) -> 'UnifiedModel':
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {'.txt', '.model'} or model_type == 'lgbm':
            meta_path = path.with_suffix(path.suffix + '.meta.json')
            params: dict[str, Any] = {}
            if meta_path.exists():
                with meta_path.open('r', encoding='utf-8') as fin:
                    meta = json.load(fin)
                model_type = model_type or meta.get('model_type', 'lgbm')
                problem_type = problem_type or meta.get('problem_type')
                params = meta.get('params', {})
            else:
                model_type = model_type or 'lgbm'

            booster = lgb.Booster(model_file=str(path))
            if problem_type is None:
                num_class = int(booster.params.get('num_class', 1)) if booster.params else 1
                problem_type = 'multiclass' if num_class > 1 else 'binary'

            wrapper = cls(model_type=model_type, problem_type=problem_type, params=params)
            wrapper.model = booster

            if meta_path.exists():
                wrapper.threshold = float(meta.get('threshold', 0.5))
                wrapper.validation_f1 = meta.get('validation_f1')

            return wrapper

        payload = joblib.load(path)
        if not isinstance(payload, dict) or 'model' not in payload:
            raise ValueError(f'Unsupported serialized model payload in {path}')

        wrapper = cls(
            model_type=payload['model_type'],
            problem_type=payload['problem_type'],
            params=payload.get('params', {}),
        )
        wrapper.model = payload['model']
        wrapper.threshold = float(payload.get('threshold', 0.5))
        wrapper.validation_f1 = payload.get('validation_f1')
        return wrapper


def _filter_labeled_rows(x, y):
    if y.ndim == 1:
        keep = y != -1
        if np.all(keep):
            return x, y
        return x[keep], y[keep]

    keep = np.sum(y, axis=1) > 0
    if np.all(keep):
        return x, y
    return x[keep], y[keep]


def _maybe_sample(x, y, sample_size: int | None, random_state: int = 42):
    if sample_size is None or sample_size >= len(y):
        return x, y

    idx = np.arange(len(y))
    if y.ndim == 1 and len(np.unique(y)) > 1:
        selected, _ = train_test_split(
            idx,
            train_size=sample_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(idx, size=sample_size, replace=False)

    selected = np.sort(selected)
    return x[selected], y[selected]


def _make_sample_weight(y, mode: str | None = None, max_weight: float = 20.0):
    """Create per-sample class weights for imbalanced binary/multiclass data.

    Supported modes:
      - None / "none": no sample weighting
      - "balanced": n_samples / (n_classes * class_count)
      - "balanced_sqrt": sqrt(balanced), safer for extremely imbalanced family data
    """
    if mode is None or str(mode).lower() in {"", "none", "false", "0"}:
        return None

    mode = str(mode).lower()
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        return None

    classes, counts = np.unique(y, return_counts=True)
    n_samples = float(len(y))
    n_classes = float(len(classes))
    weight_by_class = {
        int(cls): n_samples / (n_classes * float(cnt))
        for cls, cnt in zip(classes, counts)
    }

    sample_weight = np.asarray(
        [weight_by_class[int(label)] for label in y],
        dtype=np.float32,
    )

    if mode == "balanced_sqrt":
        sample_weight = np.sqrt(sample_weight)
    elif mode != "balanced":
        raise ValueError(
            f"Unsupported class_weight_mode={mode}. Use: none, balanced_sqrt, balanced"
        )

    if max_weight is not None and max_weight > 0:
        sample_weight = np.clip(sample_weight, 1.0 / max_weight, max_weight)

    return sample_weight




def _make_class_weight_lookup(y, mode: str | None = None, max_weight: float = 20.0):
    """Return class->weight dict for chunked/incremental training."""
    weights = _make_sample_weight(y, mode=mode, max_weight=max_weight)
    if weights is None:
        return None
    y = np.asarray(y)
    lookup: dict[int, float] = {}
    for label, weight in zip(y, weights):
        lookup[int(label)] = float(weight)
    return lookup


def _weights_for_labels(y_chunk, weight_lookup):
    if weight_lookup is None:
        return None
    return np.asarray([weight_lookup.get(int(label), 1.0) for label in y_chunk], dtype=np.float32)


def _iter_index_chunks(indices: np.ndarray, chunk_size: int):
    for start in range(0, len(indices), chunk_size):
        yield indices[start:start + chunk_size]


def _train_streaming_sgd_classifier(
    x_raw,
    y_raw,
    model_type: str,
    params: dict[str, Any] | None,
    problem_type: str,
    random_state: int,
    sample_size: int | None = None,
) -> UnifiedModel:
    """Train SGDClassifier incrementally on memmap-backed data.

    This avoids the main RAM problem of sklearn Pipeline/Ridge/SGD full-fit:
    train/validation split copies + StandardScaler full transform + solver buffers.
    Only one chunk of X is materialized at a time.
    """
    if problem_type != 'multiclass':
        raise ValueError('streaming SGD path is currently intended for Layer 2 multiclass training only.')

    fit_params = dict(params or {})
    epochs = int(fit_params.pop('epochs', fit_params.pop('max_iter', 4)))
    chunk_size = int(fit_params.pop('chunk_size', os.getenv('THREMBER_TRAIN_CHUNK_SIZE', 8192)))
    class_weight_mode = fit_params.pop('class_weight_mode', 'balanced_sqrt')
    class_weight_max = float(fit_params.pop('class_weight_max', 20.0))

    # partial_fit does not support class_weight='balanced'. Use explicit sample_weight per chunk.
    fit_params.pop('class_weight', None)
    fit_params.setdefault('random_state', random_state)
    fit_params.setdefault('average', True)

    keep_idx = np.flatnonzero(y_raw != -1).astype(np.int64, copy=False)
    if len(keep_idx) == 0:
        raise ValueError('No labeled rows found in training data.')

    if sample_size is not None and sample_size < len(keep_idx):
        y_keep = np.asarray(y_raw[keep_idx], dtype=np.int32)
        if len(np.unique(y_keep)) > 1:
            keep_idx, _ = train_test_split(
                keep_idx,
                train_size=sample_size,
                random_state=random_state,
                stratify=y_keep,
            )
        else:
            rng = np.random.default_rng(random_state)
            keep_idx = rng.choice(keep_idx, size=sample_size, replace=False)
        keep_idx = np.asarray(keep_idx, dtype=np.int64)

    y_train_all = np.asarray(y_raw[keep_idx], dtype=np.int32)
    classes = np.unique(y_train_all).astype(np.int32)
    weight_lookup = _make_class_weight_lookup(
        y_train_all,
        mode=class_weight_mode,
        max_weight=class_weight_max,
    )

    _status(
        f'streaming SGD enabled | samples={len(keep_idx)} | classes={len(classes)} | '
        f'epochs={epochs} | chunk_size={chunk_size} | weight_mode={class_weight_mode}'
    )

    scaler = StandardScaler(with_mean=False, copy=False)

    # Pass 1: fit scaler incrementally. Work on sorted indices for faster memmap reads.
    sorted_idx = np.sort(keep_idx)
    _status('streaming SGD scaler pass')
    for batch_no, batch_idx in enumerate(_iter_index_chunks(sorted_idx, chunk_size), start=1):
        x_chunk = np.array(x_raw[batch_idx], dtype=np.float32, copy=True)
        scaler.partial_fit(x_chunk)
        if batch_no == 1 or batch_no % 50 == 0:
            _status(f'scaler progress: {min(batch_no * chunk_size, len(sorted_idx))}/{len(sorted_idx)}')
        del x_chunk

    classifier = SGDClassifier(**fit_params)
    rng = np.random.default_rng(random_state)
    first_batch = True

    for epoch in range(1, max(1, epochs) + 1):
        epoch_idx = np.array(keep_idx, copy=True)
        rng.shuffle(epoch_idx)
        _status(f'streaming SGD epoch {epoch}/{epochs}')

        for batch_no, batch_idx in enumerate(_iter_index_chunks(epoch_idx, chunk_size), start=1):
            y_chunk = np.asarray(y_raw[batch_idx], dtype=np.int32)
            x_chunk = np.array(x_raw[batch_idx], dtype=np.float32, copy=True)
            x_chunk = scaler.transform(x_chunk, copy=False)
            sample_weight = _weights_for_labels(y_chunk, weight_lookup)

            if first_batch:
                classifier.partial_fit(x_chunk, y_chunk, classes=classes, sample_weight=sample_weight)
                first_batch = False
            else:
                classifier.partial_fit(x_chunk, y_chunk, sample_weight=sample_weight)

            if batch_no == 1 or batch_no % 50 == 0:
                done = min(batch_no * chunk_size, len(epoch_idx))
                _status(f'epoch {epoch} progress: {done}/{len(epoch_idx)}')

            del x_chunk, y_chunk, sample_weight

    wrapper = UnifiedModel(model_type=model_type, problem_type=problem_type, params=params)
    wrapper.model = {
        'kind': 'sgd_stream',
        'scaler': scaler,
        'classifier': classifier,
        'classes': classes,
    }
    return wrapper

def _train_val_split(x, y, validation_size: float, random_state: int):
    if validation_size <= 0 or len(y) < 2:
        return x, None, y, None

    test_size = max(1, int(round(len(y) * validation_size)))
    if test_size >= len(y):
        test_size = len(y) - 1
    if test_size <= 0:
        return x, None, y, None

    stratify = y if y.ndim == 1 and len(np.unique(y)) > 1 else None
    try:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def train_classifier(
    data_dir: Path | str,
    model_type: str,
    params: dict[str, Any] | None = None,
    problem_type: str | None = None,
    validation_size: float = 0.1,
    random_state: int = 42,
    sample_size: int | None = None,
) -> UnifiedModel:
    """Train one binary or multiclass model."""
    _status(f'load train data: {data_dir}')
    x, y = read_vectorized_features(data_dir, 'train')

    if y.ndim != 1:
        raise ValueError('train_classifier expects 1D labels. Use train_multilabel_ovr for multilabel tasks.')
    if len(y) == 0:
        raise ValueError('No rows found in training data.')

    if problem_type is None:
        labeled_y = y[y != -1]
        if len(labeled_y) == 0:
            raise ValueError('No labeled rows found in training data.')
        num_classes = int(np.max(labeled_y) + 1)
        problem_type = 'binary' if num_classes <= 2 else 'multiclass'

    # Full-data Layer 2 SGD models must use streaming partial_fit.
    # The normal sklearn Pipeline path can duplicate tens/hundreds of GB.
    if model_type.lower() in {'sgd_logreg', 'sgd_huber', 'sgd_hinge'} and problem_type == 'multiclass':
        return _train_streaming_sgd_classifier(
            x_raw=x,
            y_raw=y,
            model_type=model_type,
            params=params,
            problem_type=problem_type,
            random_state=random_state,
            sample_size=sample_size,
        )

    x, y = _filter_labeled_rows(x, y)
    if len(y) == 0:
        raise ValueError('No labeled rows found in training data.')

    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)
    _status(f'train samples: {len(y)}')

    fit_params = dict(params or {})
    class_weight_mode = fit_params.pop('class_weight_mode', None)
    class_weight_max = float(fit_params.pop('class_weight_max', 20.0))

    x_train, x_val, y_train, y_val = _train_val_split(x, y, validation_size, random_state)

    sample_weight = _make_sample_weight(
        y_train,
        mode=class_weight_mode,
        max_weight=class_weight_max,
    )
    val_sample_weight = None
    if y_val is not None:
        val_sample_weight = _make_sample_weight(
            y_val,
            mode=class_weight_mode,
            max_weight=class_weight_max,
        )

    if sample_weight is not None:
        _status(
            'sample weight enabled | '
            f'mode={class_weight_mode} | '
            f'min={float(np.min(sample_weight)):.4f} | '
            f'max={float(np.max(sample_weight)):.4f} | '
            f'mean={float(np.mean(sample_weight)):.4f}'
        )

    del x, y
    gc.collect()

    _status(f'fit model: {model_type} ({problem_type})')
    model = UnifiedModel(model_type=model_type, problem_type=problem_type, params=fit_params)
    model.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        sample_weight=sample_weight,
        val_sample_weight=val_sample_weight,
    )
    _status('fit done')
    return model


def train_multilabel_ovr(
    data_dir: Path | str,
    base_model_type: str = 'lgbm',
    params: dict[str, Any] | None = None,
    validation_size: float = 0.1,
    random_state: int = 42,
    sample_size: int | None = None,
) -> list[UnifiedModel]:
    """Train one-vs-rest models for multilabel tasks."""
    _status(f'load train data: {data_dir}')
    x, y = read_vectorized_features(data_dir, 'train')
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 2:
        raise ValueError('train_multilabel_ovr expects 2D multilabel targets.')

    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)
    _status(f'train samples: {len(y)}')

    models: list[UnifiedModel] = []
    for col in range(y.shape[1]):
        y_col = y[:, col]
        if len(np.unique(y_col)) < 2:
            continue

        x_train, x_val, y_train, y_val = _train_val_split(x, y_col, validation_size, random_state)
        model = UnifiedModel(
            model_type=base_model_type,
            problem_type='binary',
            params=params,
        )
        model.fit(x_train, y_train, x_val, y_val)
        models.append(model)

    return models


def predict_scores(model: UnifiedModel, x):
    return model.predict_scores(x)


def predict_labels(model: UnifiedModel, x, threshold: float | None = None):
    return model.predict_labels(x, threshold=threshold)


def predict_file(model: UnifiedModel, file_data: bytes, threshold: float | None = None):
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32).reshape(1, -1)
    scores = model.predict_scores(features)
    labels = model.predict_labels(features, threshold=threshold)
    return {'scores': np.asarray(scores).tolist(), 'labels': np.asarray(labels).tolist()}


def save_model(model: UnifiedModel, path: Path | str) -> None:
    model.save(path)


def load_model(path: Path | str, model_type: str | None = None, problem_type: str | None = None) -> UnifiedModel:
    return UnifiedModel.load(path, model_type=model_type, problem_type=problem_type)


def save_model_list(models: list[UnifiedModel], output_dir: Path | str, prefix: str = 'ovr') -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, model in enumerate(models):
        ext = '.txt' if model.model_type == 'lgbm' else '.joblib'
        path = output_dir / f'{prefix}_{i}{ext}'
        model.save(path)
        manifest.append(str(path.name))

    with (output_dir / f'{prefix}_manifest.json').open('w', encoding='utf-8') as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=False)


def load_model_list(output_dir: Path | str, prefix: str = 'ovr') -> list[UnifiedModel]:
    output_dir = Path(output_dir)
    with (output_dir / f'{prefix}_manifest.json').open('r', encoding='utf-8') as fin:
        manifest = json.load(fin)
    return [UnifiedModel.load(output_dir / file_name) for file_name in manifest]


def _multiclass_topk_correct(scores: np.ndarray, y_true: np.ndarray, k: int, classes=None) -> int:
    """Count top-k hits for one score chunk without storing all predictions globally."""
    if scores.ndim != 2:
        return 0
    n_classes = scores.shape[1]
    if n_classes == 0:
        return 0
    k = min(int(k), n_classes)
    classes_arr = None if classes is None else np.asarray(classes, dtype=np.int32)

    if k <= 1:
        idx = np.argmax(scores, axis=1).astype(np.int32)
        pred = classes_arr[idx] if classes_arr is not None and len(classes_arr) == n_classes else idx
        return int(np.sum(pred.astype(np.int32) == y_true))

    # For large class counts, do argpartition on small chunks only.
    topk = np.argpartition(scores, n_classes - k, axis=1)[:, -k:]
    if classes_arr is not None and len(classes_arr) == n_classes:
        topk = classes_arr[topk]
    return int(np.sum(np.any(topk == y_true.reshape(-1, 1), axis=1)))


def _eval_chunk_size(default: int = 4096) -> int:
    value = os.getenv('THREMBER_EVAL_CHUNK_SIZE')
    if value:
        return max(1, int(value))
    return default


def evaluate_classifier(
    model: UnifiedModel,
    data_dir: Path | str,
    subset: str = 'test',
    threshold: float | None = None,
) -> dict[str, Any]:
    _status(f'load eval data: {data_dir} [{subset}]')
    x_raw, y_raw = read_vectorized_features(data_dir, subset)
    total_rows = int(len(y_raw))
    x, y = _filter_labeled_rows(x_raw, y_raw)

    if y.ndim != 1:
        raise ValueError('evaluate_classifier expects 1D labels.')
    if len(y) == 0:
        raise ValueError(f'No labeled rows found in subset={subset}.')

    _status('predict scores')

    if model.problem_type == 'binary':
        scores = np.asarray(model.predict_scores(x))
        used_threshold = threshold
        if used_threshold is None:
            used_threshold = getattr(model, 'threshold', 0.5)

        labels = (scores >= used_threshold).astype(np.int32)

        roc_auc = float(roc_auc_score(y, scores))
        pr_auc = float(average_precision_score(y, scores))

        fpr, tpr, _ = roc_curve(y, scores)
        tpr_at_1pct = _tpr_at_max_fpr(fpr, tpr, max_fpr=0.01)

        best_eval_threshold, best_eval_f1 = _best_binary_threshold_by_f1(y, scores)

        return {
            'eval_rows_total_before_y_filter': total_rows,
            'eval_rows_labeled_after_y_filter': int(len(y)),
            'labeled_row_fraction': float(len(y) / max(total_rows, 1)),

            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tpr_at_fpr_1pct': float(tpr_at_1pct),

            'threshold': float(used_threshold),
            'accuracy': float(np.mean(labels == y)),
            'f1': float(f1_score(y, labels, zero_division=0)),

            'validation_threshold': float(getattr(model, 'threshold', 0.5)),
            'validation_f1': (
                None
                if getattr(model, 'validation_f1', None) is None
                else float(model.validation_f1)
            ),

            'best_eval_threshold_by_f1': float(best_eval_threshold),
            'best_eval_f1': float(best_eval_f1),
        }

    # Multiclass evaluation is chunked to avoid holding a huge
    # [num_samples, num_classes] score matrix in RAM.
    n = int(len(y))
    labels = np.empty(n, dtype=np.int32)
    chunk_size = _eval_chunk_size()
    topk_values = (3, 5, 10)
    topk_hits = {k: 0 for k in topk_values}
    score_classes = model.score_classes()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores_chunk = np.asarray(model.predict_scores(x[start:end]))

        if scores_chunk.ndim == 1:
            labels[start:end] = scores_chunk.astype(np.int32)
            continue

        pred_idx = np.argmax(scores_chunk, axis=1).astype(np.int32)
        if score_classes is not None and len(score_classes) == scores_chunk.shape[1]:
            labels[start:end] = np.asarray(score_classes, dtype=np.int32)[pred_idx]
        else:
            labels[start:end] = pred_idx
        y_chunk = y[start:end].astype(np.int32, copy=False)
        for k in topk_values:
            topk_hits[k] += _multiclass_topk_correct(scores_chunk, y_chunk, k, classes=score_classes)

        if start == 0 or end == n or ((start // chunk_size) % 20 == 0):
            _status(f'eval progress: {end}/{n}')

    unique_true = int(len(np.unique(y)))
    unique_pred = int(len(np.unique(labels)))

    metrics = {
        'eval_rows_total_before_y_filter': total_rows,
        'eval_rows_labeled_after_y_filter': n,
        'labeled_row_fraction': float(n / max(total_rows, 1)),
        'num_true_classes_in_eval': unique_true,
        'num_predicted_classes_in_eval': unique_pred,
        'accuracy': float(accuracy_score(y, labels)),
        'f1_macro': float(f1_score(y, labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, labels, average='weighted', zero_division=0)),
    }

    for k, hits in topk_hits.items():
        metrics[f'top_{k}_accuracy'] = float(hits / max(n, 1))

    return metrics

def evaluate_multilabel_ovr(
    models: list[UnifiedModel],
    data_dir: Path | str,
    subset: str = 'test',
    threshold: float = 0.5,
) -> dict[str, float]:
    _status(f'load eval data: {data_dir} [{subset}]')
    x, y = read_vectorized_features(data_dir, subset)
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 2:
        raise ValueError('evaluate_multilabel_ovr expects 2D multilabel targets.')
    if not models:
        raise ValueError('evaluate_multilabel_ovr received an empty model list.')

    pred_cols = []
    for model in models:
        scores = model.predict_scores(x)
        pred_cols.append((np.asarray(scores) >= threshold).astype(np.int32))

    y_pred = np.stack(pred_cols, axis=1)
    y_true = y[:, : y_pred.shape[1]]

    return {
        'exact_match': float(np.mean(np.all(y_pred == y_true, axis=1))),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def make_binary_auc_scorer():
    return make_scorer(roc_auc_score, needs_proba=True, max_fpr=5e-3)


def tune_classifier(
    data_dir: Path | str,
    model_type: str = 'lgbm',
    problem_type: str = 'binary',
    sample_size: int | None = 200000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Basic hyperparameter tuning helper for binary LightGBM."""
    if model_type != 'lgbm':
        raise ValueError("tune_classifier currently supports only model_type='lgbm'.")
    if problem_type != 'binary':
        raise ValueError("tune_classifier currently supports only problem_type='binary'.")

    x, y = read_vectorized_features(data_dir, 'train')
    x, y = _filter_labeled_rows(x, y)
    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)

    progressive_cv = TimeSeriesSplit(n_splits=3).split(x)

    estimator = lgb.LGBMClassifier(n_jobs=-1, verbose=-1)
    param_grid = {
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'n_estimators': [300, 500],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [64, 128, 256],
        'feature_fraction': [0.8, 1.0],
        'bagging_fraction': [0.8, 1.0],
    }

    grid = GridSearchCV(
        estimator=estimator,
        cv=progressive_cv,
        param_grid=param_grid,
        scoring=make_binary_auc_scorer(),
        n_jobs=1,
        verbose=2,
    )
    grid.fit(x, y, categorical_feature=LGBM_CATEGORICAL_FEATURES)
    return grid.best_params_
