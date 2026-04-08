from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from .dataset import read_vectorized_features
from .features import PEFeatureExtractor


LGBM_CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


@dataclass
class FitConfig:
    model_type: str
    problem_type: str
    validation_size: float = 0.1
    random_state: int = 42
    sample_size: int | None = None
    params: dict[str, Any] | None = None


class UnifiedModel:
    """
    One wrapper class for all supported model families.
    """

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

    def _build_sklearn_model(self):
        if self.model_type == "logreg":
            return LogisticRegression(**self.params)
        if self.model_type == "rf":
            return RandomForestClassifier(**self.params)
        if self.model_type == "mlp":
            return MLPClassifier(**self.params)
        raise ValueError(f"Unsupported sklearn model_type: {self.model_type}")

    def fit(self, x_train, y_train, x_val=None, y_val=None) -> "UnifiedModel":
        if self.model_type == "lgbm":
            self.model = self._fit_lightgbm(x_train, y_train, x_val, y_val)
        else:
            self.model = self._build_sklearn_model()
            self.model.fit(x_train, y_train)
        return self

    def _fit_lightgbm(self, x_train, y_train, x_val=None, y_val=None):
        params = dict(self.params)

        if self.problem_type == "binary":
            params.setdefault("objective", "binary")
            params.setdefault("metric", ["auc", "binary_logloss"])
        elif self.problem_type == "multiclass":
            params.setdefault("objective", "multiclass")
            params["num_class"] = int(np.max(y_train) + 1)
            params.setdefault("metric", "multi_logloss")
        else:
            raise ValueError(f"LightGBM in UnifiedModel supports binary/multiclass only, got {self.problem_type}")

        train_set = lgb.Dataset(
            x_train,
            y_train,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )

        valid_sets = None
        if x_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                x_val,
                y_val,
                reference=train_set,
                categorical_feature=self.categorical_features,
                free_raw_data=False,
            )
            valid_sets = [val_set]

        return lgb.train(params, train_set, valid_sets=valid_sets)

    def _ensure_fitted(self):
        if self.model is None:
            raise ValueError("Model is not fitted or loaded yet.")

    def predict_scores(self, x):
        self._ensure_fitted()

        if self.model_type == "lgbm":
            return self.model.predict(x)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)
            if self.problem_type == "binary":
                return probs[:, 1]
            return probs

        return self.model.predict(x)

    def predict_labels(self, x, threshold: float = 0.5):
        scores = self.predict_scores(x)

        if self.problem_type == "binary":
            return (np.asarray(scores) >= threshold).astype(np.int32)

        if self.problem_type == "multiclass":
            if np.asarray(scores).ndim == 1:
                return np.asarray(scores).astype(np.int32)
            return np.argmax(scores, axis=1).astype(np.int32)

        if self.problem_type == "multilabel":
            scores = np.asarray(scores)
            return (scores >= threshold).astype(np.int32)

        raise ValueError(f"Unsupported problem_type: {self.problem_type}")

    def save(self, path: Path | str) -> None:
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == "lgbm":
            self.model.save_model(str(path))
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            with meta_path.open("w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "model_type": self.model_type,
                        "problem_type": self.problem_type,
                        "params": self.params,
                    },
                    fout,
                    indent=2,
                    ensure_ascii=False,
                )
            return

        joblib.dump(
            {
                "model_type": self.model_type,
                "problem_type": self.problem_type,
                "params": self.params,
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, model_type: str | None = None, problem_type: str | None = None) -> "UnifiedModel":
        path = Path(path)

        if path.suffix == ".txt" or model_type == "lgbm":
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            if model_type is None or problem_type is None:
                if not meta_path.exists():
                    raise ValueError("Missing LightGBM meta file. Please pass model_type and problem_type manually.")
                with meta_path.open("r", encoding="utf-8") as fin:
                    meta = json.load(fin)
                model_type = meta["model_type"]
                problem_type = meta["problem_type"]
                params = meta.get("params", {})
            else:
                params = {}

            wrapper = cls(model_type=model_type, problem_type=problem_type, params=params)
            wrapper.model = lgb.Booster(model_file=str(path))
            return wrapper

        payload = joblib.load(path)
        wrapper = cls(
            model_type=payload["model_type"],
            problem_type=payload["problem_type"],
            params=payload.get("params", {}),
        )
        wrapper.model = payload["model"]
        return wrapper


def _filter_labeled_rows(x, y):
    if y.ndim == 1:
        keep = y != -1
        return x[keep], y[keep]
    keep = np.sum(y, axis=1) > 0
    return x[keep], y[keep]


def _maybe_sample(x, y, sample_size: int | None, random_state: int = 42):
    if sample_size is None or sample_size >= len(y):
        return x, y

    idx = np.arange(len(y))
    if y.ndim == 1:
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


def train_classifier(
    data_dir: Path | str,
    model_type: str,
    params: dict[str, Any] | None = None,
    problem_type: str | None = None,
    validation_size: float = 0.1,
    random_state: int = 42,
    sample_size: int | None = None,
) -> UnifiedModel:
    """
    Train one binary or multiclass model.
    """
    x, y = read_vectorized_features(data_dir, "train")
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 1:
        raise ValueError("train_classifier expects 1D labels. Use train_multilabel_ovr for multilabel tasks.")

    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)

    if problem_type is None:
        num_classes = int(np.max(y) + 1)
        problem_type = "binary" if num_classes <= 2 else "multiclass"

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )

    model = UnifiedModel(model_type=model_type, problem_type=problem_type, params=params)
    model.fit(x_train, y_train, x_val, y_val)
    return model


def train_multilabel_ovr(
    data_dir: Path | str,
    base_model_type: str = "lgbm",
    params: dict[str, Any] | None = None,
    validation_size: float = 0.1,
    random_state: int = 42,
    sample_size: int | None = None,
) -> list[UnifiedModel]:
    """
    Train one-vs-rest models for multilabel tasks.
    """
    x, y = read_vectorized_features(data_dir, "train")
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 2:
        raise ValueError("train_multilabel_ovr expects 2D multilabel targets.")

    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)

    models: list[UnifiedModel] = []
    for col in range(y.shape[1]):
        y_col = y[:, col]
        if len(np.unique(y_col)) < 2:
            continue

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y_col,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_col,
        )

        model = UnifiedModel(
            model_type=base_model_type,
            problem_type="binary",
            params=params,
        )
        model.fit(x_train, y_train, x_val, y_val)
        models.append(model)

    return models


def predict_scores(model: UnifiedModel, x):
    return model.predict_scores(x)


def predict_labels(model: UnifiedModel, x, threshold: float = 0.5):
    return model.predict_labels(x, threshold=threshold)


def predict_file(model: UnifiedModel, file_data: bytes, threshold: float = 0.5):
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32).reshape(1, -1)

    scores = model.predict_scores(features)
    labels = model.predict_labels(features, threshold=threshold)

    return {
        "scores": np.asarray(scores).tolist(),
        "labels": np.asarray(labels).tolist(),
    }


def save_model(model: UnifiedModel, path: Path | str) -> None:
    model.save(path)


def load_model(path: Path | str, model_type: str | None = None, problem_type: str | None = None) -> UnifiedModel:
    return UnifiedModel.load(path, model_type=model_type, problem_type=problem_type)


def save_model_list(models: list[UnifiedModel], output_dir: Path | str, prefix: str = "ovr") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, model in enumerate(models):
        ext = ".txt" if model.model_type == "lgbm" else ".joblib"
        path = output_dir / f"{prefix}_{i}{ext}"
        model.save(path)
        manifest.append(str(path.name))

    with (output_dir / f"{prefix}_manifest.json").open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=False)


def load_model_list(output_dir: Path | str, prefix: str = "ovr") -> list[UnifiedModel]:
    output_dir = Path(output_dir)
    with (output_dir / f"{prefix}_manifest.json").open("r", encoding="utf-8") as fin:
        manifest = json.load(fin)
    return [UnifiedModel.load(output_dir / file_name) for file_name in manifest]


def evaluate_classifier(
    model: UnifiedModel,
    data_dir: Path | str,
    subset: str = "test",
    threshold: float = 0.5,
) -> dict[str, float]:
    x, y = read_vectorized_features(data_dir, subset)
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 1:
        raise ValueError("evaluate_classifier expects 1D labels.")

    scores = model.predict_scores(x)

    if model.problem_type == "binary":
        labels = (np.asarray(scores) >= threshold).astype(np.int32)
        roc_auc = float(roc_auc_score(y, scores))
        pr_auc = float(average_precision_score(y, scores))
        fpr, tpr, _ = roc_curve(y, scores)
        idx = int(np.argmin(np.abs(fpr - 0.01)))
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tpr_at_fpr_1pct": float(tpr[idx]),
            "accuracy": float(np.mean(labels == y)),
            "f1": float(f1_score(y, labels)),
        }

    labels = model.predict_labels(x)
    return {
        "accuracy": float(accuracy_score(y, labels)),
        "f1_macro": float(f1_score(y, labels, average="macro")),
        "f1_weighted": float(f1_score(y, labels, average="weighted")),
    }


def evaluate_multilabel_ovr(
    models: list[UnifiedModel],
    data_dir: Path | str,
    subset: str = "test",
    threshold: float = 0.5,
) -> dict[str, float]:
    x, y = read_vectorized_features(data_dir, subset)
    x, y = _filter_labeled_rows(x, y)

    if y.ndim != 2:
        raise ValueError("evaluate_multilabel_ovr expects 2D multilabel targets.")

    pred_cols = []
    for model in models:
        scores = model.predict_scores(x)
        pred_cols.append((np.asarray(scores) >= threshold).astype(np.int32))

    y_pred = np.stack(pred_cols, axis=1)
    y_true = y[:, : y_pred.shape[1]]

    exact_match = float(np.mean(np.all(y_pred == y_true, axis=1)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "exact_match": exact_match,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
    }


def make_binary_auc_scorer():
    return make_scorer(roc_auc_score, needs_proba=True, max_fpr=5e-3)


def tune_classifier(
    data_dir: Path | str,
    model_type: str = "lgbm",
    problem_type: str = "binary",
    sample_size: int | None = 200000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Basic hyperparameter tuning helper.

    Right now this is intended for LightGBM on binary classification,
    similar to the old optimize_model() role.
    """
    if model_type != "lgbm":
        raise ValueError("tune_classifier currently supports only model_type='lgbm'.")
    if problem_type != "binary":
        raise ValueError("tune_classifier currently supports only problem_type='binary'.")

    x, y = read_vectorized_features(data_dir, "train")
    x, y = _filter_labeled_rows(x, y)
    x, y = _maybe_sample(x, y, sample_size, random_state=random_state)

    progressive_cv = TimeSeriesSplit(n_splits=3).split(x)

    estimator = lgb.LGBMClassifier(n_jobs=-1, verbose=-1)
    param_grid = {
        "boosting_type": ["gbdt"],
        "objective": ["binary"],
        "n_estimators": [300, 500],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [64, 128, 256],
        "feature_fraction": [0.8, 1.0],
        "bagging_fraction": [0.8, 1.0],
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
