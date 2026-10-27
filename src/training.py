"""Training, evaluation and ensemble-introspection helpers (no Streamlit here)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .data import Dataset, SplitData, split_data
from .models import (
    CATEGORY, STACKING_MEMBERS, VOTING_MEMBERS, VOTING, XGB,
    build_model, validate_for_data, validate_params,
)


@dataclass
class ModelResult:
    """Everything we keep about one trained model."""

    name: str
    category: str
    params: dict
    model: Any
    train_time: float
    y_pred: np.ndarray
    metrics: dict
    confusion: np.ndarray


# ------------------------------------------------------------ evaluation
def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus weighted precision / recall / F1 (works for 2+ classes)."""
    kw = dict(average="weighted", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, **kw)),
        "Recall": float(recall_score(y_true, y_pred, **kw)),
        "F1-score": float(f1_score(y_true, y_pred, **kw)),
    }


def train_and_evaluate(name: str, params: dict | None, split: SplitData, seed: int = 42) -> ModelResult:
    """Fit ONE model on the training split and score it on the test split.

    This is the single place where models are trained, so every page (playground,
    worked example, tests...) uses exactly the same methodology.
    """
    clean = validate_params(name, params)
    validate_for_data(name, clean, split.y_train)
    model = build_model(name, clean, seed)
    start = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    train_time = time.perf_counter() - start
    y_pred = model.predict(split.X_test)
    labels = np.arange(len(split.class_names))
    return ModelResult(
        name=name, category=CATEGORY[name], params=clean, model=model, train_time=train_time,
        y_pred=y_pred, metrics=compute_metrics(split.y_test, y_pred),
        confusion=confusion_matrix(split.y_test, y_pred, labels=labels),
    )


def run_experiment(selected: dict, split: SplitData, seed: int = 42):
    """Train several models; a failing model is reported but never stops the others.

    ``selected`` maps algorithm name -> parameter dict. Returns ``(results, errors)``.
    """
    results, errors = [], {}
    for name, params in selected.items():
        try:
            results.append(train_and_evaluate(name, params, split, seed))
        except Exception as exc:  # noqa: BLE001 - we want to show any failure to the user
            errors[name] = str(exc)
    return results, errors


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """One row per model, sorted by accuracy (best first)."""
    rows = [
        {"Algorithm": r.name, "Type": r.category, **r.metrics, "Train time (s)": r.train_time}
        for r in results
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["Accuracy", "Train time (s)"], ascending=[False, True]).reset_index(drop=True)


def cross_validate_model(name: str, params: dict | None, split: SplitData, seed: int = 42, folds: int = 5):
    """Cross-validated accuracy on the TRAINING data only (mean, std)."""
    clean = validate_params(name, params)
    validate_for_data(name, clean, split.y_train)
    folds = int(min(folds, np.bincount(split.y_train).min()))
    cv = StratifiedKFold(n_splits=max(folds, 2), shuffle=True, random_state=seed)
    scores = cross_val_score(build_model(name, clean, seed), split.X_train, split.y_train, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def multi_seed_comparison(selected: dict, ds: Dataset, test_size: float, seeds: list[int]) -> pd.DataFrame:
    """Re-run the same models on several different splits (robustness check)."""
    rows = []
    for s in seeds:
        split = split_data(ds, test_size, s)
        results, _ = run_experiment(selected, split, s)
        rows += [{"Algorithm": r.name, "Seed": s, "Accuracy": r.metrics["Accuracy"]} for r in results]
    return pd.DataFrame(rows)


def pairwise_agreement(results: list[ModelResult]) -> pd.DataFrame:
    """Share of test samples on which each pair of models predicts the same class."""
    names = [r.name for r in results]
    mat = np.array([[float(np.mean(a.y_pred == b.y_pred)) for b in results] for a in results])
    return pd.DataFrame(mat, index=names, columns=names)


# ------------------------------------------------------ boosting curves
def supports_staged(model) -> bool:
    return hasattr(model, "staged_predict") or type(model).__name__ == "XGBClassifier"


def staged_scores(model, split: SplitData) -> pd.DataFrame | None:
    """Accuracy after 1, 2, 3 ... boosting iterations (train and test)."""
    if hasattr(model, "staged_predict"):
        train = [accuracy_score(split.y_train, p) for p in model.staged_predict(split.X_train)]
        test = [accuracy_score(split.y_test, p) for p in model.staged_predict(split.X_test)]
    elif type(model).__name__ == "XGBClassifier":
        n = int(model.get_params()["n_estimators"])
        train = [accuracy_score(split.y_train, model.predict(split.X_train, iteration_range=(0, i))) for i in range(1, n + 1)]
        test = [accuracy_score(split.y_test, model.predict(split.X_test, iteration_range=(0, i))) for i in range(1, n + 1)]
    else:
        return None
    return pd.DataFrame({"Iteration": np.arange(1, len(test) + 1), "Train accuracy": train, "Test accuracy": test})


# ------------------------------------------------- ensemble introspection
def is_member_ensemble(model) -> bool:
    return isinstance(model, (VotingClassifier, StackingClassifier, BaggingClassifier, RandomForestClassifier))


def get_members(model, max_members: int = 8) -> list[tuple[str, Any]]:
    """Fitted sub-models of a Voting / Stacking / Bagging / Random Forest ensemble."""
    if isinstance(model, VotingClassifier):
        return [(VOTING_MEMBERS[k], est) for k, est in zip(model.named_estimators_.keys(), model.estimators_)]
    if isinstance(model, StackingClassifier):
        return [(STACKING_MEMBERS[k], est) for k, est in zip(model.named_estimators_.keys(), model.estimators_)]
    if isinstance(model, (BaggingClassifier, RandomForestClassifier)):
        return [(f"Tree {i + 1}", est) for i, est in enumerate(model.estimators_[:max_members])]
    return []


def _member_labels(model, est, X) -> np.ndarray:
    """Predicted class *indices* from a member (members are trained on encoded labels)."""
    return np.asarray(est.predict(X)).astype(int)


def member_prediction_table(model, X, y_true, class_names, n_samples: int = 10, max_members: int = 7) -> pd.DataFrame:
    """Rows = first ``n_samples`` test samples; columns = each member + final ensemble answer."""
    members = get_members(model, max_members)
    X = X[:n_samples]
    names = np.array(class_names)
    data = {"True label": names[np.asarray(y_true[:n_samples]).astype(int)]}
    for label, est in members:
        data[label] = names[model.classes_[_member_labels(model, est, X)]]
    data["ENSEMBLE (final)"] = names[model.classes_.searchsorted(model.predict(X))]
    df = pd.DataFrame(data)
    df.index = [f"Test sample {i}" for i in range(len(df))]
    return df


def member_probabilities(model, x_row, class_names) -> pd.DataFrame:
    """Class probabilities from each Voting member for ONE sample (+ their average)."""
    x = np.asarray(x_row).reshape(1, -1)
    rows, probs = {}, []
    for label, est in get_members(model):
        p = est.predict_proba(x)[0]
        rows[label] = p
        probs.append(p)
    rows["Average (soft vote)"] = np.mean(probs, axis=0)
    return pd.DataFrame(rows, index=class_names).T


def stacking_weights(model: StackingClassifier, n_classes: int) -> pd.DataFrame:
    """How strongly the meta-model leans on each base model (mean |coefficient|)."""
    coef = np.atleast_2d(model.final_estimator_.coef_)
    n_base = len(model.estimators_)
    grouped = np.abs(coef.reshape(coef.shape[0], n_base, -1)).mean(axis=(0, 2))
    labels = [STACKING_MEMBERS[k] for k in model.named_estimators_.keys()]
    return pd.DataFrame({"Base model": labels, "Meta-model weight": grouped})


# ---------------------------------------------------- feature importance
def feature_importance(model, feature_names) -> pd.DataFrame | None:
    """Impurity-based importance for tree ensembles; ``None`` if unsupported."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    elif isinstance(model, BaggingClassifier) and all(isinstance(e, DecisionTreeClassifier) for e in model.estimators_):
        imp = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    else:
        return None
    return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False).reset_index(drop=True)


def permutation_importance_df(model, split: SplitData, seed: int = 42, n_repeats: int = 5) -> pd.DataFrame:
    """Model-agnostic importance: how much does shuffling a feature hurt TEST accuracy?"""
    r = permutation_importance(model, split.X_test, split.y_test, n_repeats=n_repeats, random_state=seed, scoring="accuracy")
    df = pd.DataFrame({"Feature": split.feature_names, "Importance": r.importances_mean, "Std": r.importances_std})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------- 2-D helpers
def fit_2d_models(selected: dict, split: SplitData, feature_idx: tuple[int, int], seed: int = 42) -> dict:
    """Re-train each selected model on just two features so a decision boundary can be drawn."""
    cols = list(feature_idx)
    fitted = {}
    for name, params in selected.items():
        try:
            clean = validate_params(name, params)
            validate_for_data(name, clean, split.y_train)
            model = build_model(name, clean, seed)
            model.fit(split.X_train[:, cols], split.y_train)
            fitted[name] = model
        except Exception:  # noqa: BLE001 - skip models that cannot train on 2 features
            continue
    return fitted


# ------------------------------------------------------ theory-page demos
def majority_vote_accuracy(p: float, n: int) -> float:
    """P(majority of n INDEPENDENT voters, each right with prob. p, is right). n must be odd."""
    from math import comb

    if n < 1 or n % 2 == 0:
        raise ValueError("n must be a positive odd number.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1.")
    return float(sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1)))


def variance_experiment(split: SplitData, n_runs: int = 15, n_estimators: int = 30, seed: int = 42) -> pd.DataFrame:
    """Train a deep tree vs a bagged forest of deep trees on 15 different 70% subsamples."""
    from sklearn.ensemble import BaggingClassifier as _Bag

    rng = np.random.RandomState(seed)
    n = len(split.y_train)
    rows = []
    for run in range(n_runs):
        idx = rng.choice(n, size=int(0.7 * n), replace=False)
        Xs, ys = split.X_train[idx], split.y_train[idx]
        tree = DecisionTreeClassifier(random_state=run).fit(Xs, ys)
        bag = _Bag(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=run).fit(Xs, ys)
        rows.append({"Model": "Single deep tree", "Run": run, "Test accuracy": tree.score(split.X_test, split.y_test)})
        rows.append({"Model": "Bagged trees", "Run": run, "Test accuracy": bag.score(split.X_test, split.y_test)})
    return pd.DataFrame(rows)


def oob_vs_test(split: SplitData, n_estimators: int = 100, seed: int = 42) -> dict:
    """Out-of-bag accuracy (free validation) vs. real test accuracy of a Random Forest."""
    rf = RandomForestClassifier(n_estimators=max(n_estimators, 30), oob_score=True, random_state=seed)
    rf.fit(split.X_train, split.y_train)
    return {"oob_accuracy": float(rf.oob_score_), "test_accuracy": float(rf.score(split.X_test, split.y_test))}


def bootstrap_demo(n: int = 12, seed: int = 0) -> dict:
    """One bootstrap sample of n row ids, plus how many distinct rows it contains."""
    rng = np.random.RandomState(seed)
    draw = rng.choice(n, size=n, replace=True)
    uniq = np.unique(draw)
    return {"draw": draw, "unique": len(uniq), "out_of_bag": sorted(set(range(n)) - set(uniq.tolist()))}
