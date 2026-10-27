"""Dataset loading and train/test splitting.

Design note (avoiding data leakage)
-----------------------------------
`split_data` returns RAW (unscaled) arrays. Any preprocessing such as scaling
lives *inside* each model's scikit-learn Pipeline (see ``models.py``), so it is
fitted on the training data only - never on the test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_moons,
)
from sklearn.model_selection import train_test_split

IRIS = "Iris"
BREAST_CANCER = "Breast Cancer"
SYNTHETIC = "Synthetic (make_classification)"
DATASET_NAMES = [IRIS, BREAST_CANCER, SYNTHETIC]


@dataclass
class Dataset:
    """A labelled dataset plus human-readable names."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    class_names: list

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass
class SplitData:
    """Train/test arrays that every model in an experiment shares."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    class_names: list


def _check_seed(seed) -> int:
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= int(seed) <= 2**31 - 1:
        raise ValueError("Random seed must be a whole number between 0 and 2,147,483,647.")
    return int(seed)


def load_dataset(
    name: str,
    n_samples: int = 500,
    n_features: int = 2,
    n_classes: int = 2,
    class_sep: float = 1.0,
    label_noise: float = 0.03,
    seed: int = 42,
) -> Dataset:
    """Load a built-in dataset or generate a synthetic one.

    The synthetic-only arguments are ignored for Iris / Breast Cancer.
    """
    seed = _check_seed(seed)
    if name == IRIS:
        d = load_iris()
        return Dataset(name, d.data, d.target, list(d.feature_names), [str(c) for c in d.target_names])
    if name == BREAST_CANCER:
        d = load_breast_cancer()
        return Dataset(name, d.data, d.target, list(d.feature_names), [str(c) for c in d.target_names])
    if name == SYNTHETIC:
        if not 100 <= n_samples <= 5000:
            raise ValueError("Number of samples must be between 100 and 5000.")
        if not 2 <= n_features <= 20:
            raise ValueError("Number of features must be between 2 and 20.")
        if n_classes not in (2, 3):
            raise ValueError("Number of classes must be 2 or 3.")
        if not 0.3 <= class_sep <= 3.0:
            raise ValueError("Class separation must be between 0.3 and 3.0.")
        if not 0.0 <= label_noise <= 0.3:
            raise ValueError("Label noise must be between 0 and 0.3.")
        # Keep make_classification's internal constraints satisfied:
        # n_classes * n_clusters_per_class <= 2 ** n_informative
        n_informative = min(n_features, max(2, n_features // 2))
        n_redundant = min(2, n_features - n_informative)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=2 if n_classes == 2 else 1,
            class_sep=class_sep,
            flip_y=label_noise,
            random_state=seed,
        )
        return Dataset(
            name, X, y,
            [f"feature_{i + 1}" for i in range(n_features)],
            [f"Class {i}" for i in range(n_classes)],
        )
    raise ValueError(f"Unknown dataset: {name!r}")


def make_demo_data(n_samples: int = 400, noise: float = 0.3, seed: int = 42) -> Dataset:
    """Small 2-D 'two moons' dataset used by the theory demos (non-linear, visual)."""
    seed = _check_seed(seed)
    if not 100 <= n_samples <= 3000:
        raise ValueError("Number of samples must be between 100 and 3000.")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return Dataset("Two Moons (2-D demo)", X, y, ["x1", "x2"], ["Class 0", "Class 1"])


def split_data(ds: Dataset, test_size: float = 0.25, seed: int = 42) -> SplitData:
    """Stratified train/test split shared by all models in an experiment."""
    seed = _check_seed(seed)
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("Test size must be between 0.1 and 0.5.")
    if np.bincount(ds.y).min() < 2:
        raise ValueError("Every class needs at least 2 samples for a stratified split.")
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            ds.X, ds.y, test_size=test_size, random_state=seed, stratify=ds.y
        )
    except ValueError as exc:  # e.g. test set smaller than number of classes
        raise ValueError(f"Could not create a stratified split: {exc}") from exc
    return SplitData(X_tr, X_te, y_tr, y_te, list(ds.feature_names), list(ds.class_names))
