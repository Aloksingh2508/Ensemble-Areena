"""Model registry: parameter specs, validation and model construction.

Every algorithm is described once by a list of ``ParamSpec`` objects. The UI
builds its sliders from these specs and ``validate_params`` uses the *same*
specs to reject bad input, so UI and validation can never drift apart.

Scale-sensitive models (Logistic Regression, KNN, SVM) are wrapped in a
``Pipeline(StandardScaler, model)``. The scaler is therefore fitted only on
whatever data ``fit`` receives (the training set) - this prevents leakage.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------- names
LOGREG = "Logistic Regression"
TREE = "Decision Tree"
KNN = "K-Nearest Neighbors"
SVM = "Support Vector Machine"
FOREST = "Random Forest"
BAGGING = "Bagging Classifier"
ADABOOST = "AdaBoost"
GRADBOOST = "Gradient Boosting"
HISTGB = "HistGradientBoosting"
VOTING = "Voting Classifier"
STACKING = "Stacking Classifier"
XGB = "XGBoost"

CATEGORY = {
    LOGREG: "Single model", TREE: "Single model", KNN: "Single model", SVM: "Single model",
    FOREST: "Bagging", BAGGING: "Bagging",
    ADABOOST: "Boosting", GRADBOOST: "Boosting", HISTGB: "Boosting", XGB: "Boosting",
    VOTING: "Voting", STACKING: "Stacking",
}

# Short keys -> readable labels for the members inside Voting / Stacking
VOTING_MEMBERS = {"lr": LOGREG, "dt": TREE, "knn": KNN}
STACKING_MEMBERS = {"dt": TREE, "lr": LOGREG, "rf": FOREST}


def xgboost_available() -> bool:
    """XGBoost is optional - the app works fine without it."""
    return importlib.util.find_spec("xgboost") is not None


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: str  # "int" | "float" | "choice"
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    options: tuple = ()
    help: str = ""


def _depth(default, lo=1, hi=20, label="Max tree depth", help_="Deeper trees can fit more detail but overfit more easily."):
    return ParamSpec("max_depth", label, "int", default, lo, hi, 1, help=help_)


def _n_est(default, lo=1, hi=300, help_="How many models are trained and combined."):
    return ParamSpec("n_estimators", "Number of estimators", "int", default, lo, hi, 5, help=help_)


PARAM_SPECS: dict[str, list[ParamSpec]] = {
    LOGREG: [ParamSpec("C", "Regularisation C", "float", 1.0, 0.1, 20.0, 0.1,
                       help="Larger C = weaker regularisation.")],
    TREE: [_depth(5), ParamSpec("min_samples_leaf", "Min samples per leaf", "int", 1, 1, 20, 1,
                                help="Larger values give smoother, simpler trees.")],
    KNN: [ParamSpec("n_neighbors", "Neighbours (k)", "int", 5, 1, 30, 1, help="How many neighbours vote."),
          ParamSpec("weights", "Vote weighting", "choice", "uniform", options=("uniform", "distance"))],
    SVM: [ParamSpec("C", "Regularisation C", "float", 1.0, 0.1, 20.0, 0.1),
          ParamSpec("kernel", "Kernel", "choice", "rbf", options=("rbf", "linear", "poly"))],
    FOREST: [_n_est(100, help_="Number of trees in the forest."), _depth(10)],
    BAGGING: [_n_est(50, 1, 200), _depth(10, label="Base tree max depth"),
              ParamSpec("max_samples", "Bootstrap sample fraction", "float", 1.0, 0.1, 1.0, 0.1,
                        help="Fraction of the training set drawn (with replacement) for each model.")],
    ADABOOST: [_n_est(50), ParamSpec("learning_rate", "Learning rate", "float", 1.0, 0.01, 2.0, 0.01,
                                     help="Shrinks each model's contribution."),
               _depth(1, 1, 5, "Weak-learner depth", "Depth 1 = a 'stump', the classic AdaBoost weak learner.")],
    GRADBOOST: [_n_est(100), ParamSpec("learning_rate", "Learning rate", "float", 0.1, 0.01, 1.0, 0.01,
                                       help="Smaller values need more trees but generalise better."),
                _depth(3, 1, 6)],
    HISTGB: [ParamSpec("max_iter", "Boosting iterations", "int", 100, 10, 300, 10),
             ParamSpec("learning_rate", "Learning rate", "float", 0.1, 0.01, 1.0, 0.01),
             _depth(5, 1, 10)],
    VOTING: [ParamSpec("voting", "Voting type", "choice", "soft", options=("hard", "soft"),
                       help="hard = majority of labels, soft = average of probabilities."),
             _depth(5, label="Tree member max depth"),
             ParamSpec("n_neighbors", "KNN member neighbours", "int", 5, 1, 30, 1)],
    STACKING: [ParamSpec("cv", "Out-of-fold CV folds", "int", 5, 2, 10, 1,
                         help="Folds used to create out-of-fold predictions for the meta-model."),
               ParamSpec("rf_estimators", "Forest member trees", "int", 50, 10, 200, 10),
               _depth(5, label="Tree member max depth")],
    XGB: [_n_est(100), ParamSpec("learning_rate", "Learning rate", "float", 0.1, 0.01, 1.0, 0.01),
          _depth(3, 1, 10)],
}


def available_algorithms() -> list[str]:
    """Algorithms offered in the UI (the 10 required ones + extras)."""
    names = [LOGREG, TREE, KNN, SVM, FOREST, BAGGING, ADABOOST, GRADBOOST, HISTGB, VOTING, STACKING]
    if xgboost_available():
        names.append(XGB)
    return names


def get_default_params(name: str) -> dict:
    return {s.key: s.default for s in PARAM_SPECS[name]}


def validate_params(name: str, params: dict | None = None) -> dict:
    """Return a clean, fully-populated parameter dict or raise ``ValueError``."""
    if name not in PARAM_SPECS:
        raise ValueError(f"Unknown algorithm: {name!r}")
    params = dict(params or {})
    specs = {s.key: s for s in PARAM_SPECS[name]}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}")
    clean = {}
    for key, spec in specs.items():
        value = params.get(key, spec.default)
        if spec.kind == "choice":
            if value not in spec.options:
                raise ValueError(f"{name}: '{spec.label}' must be one of {list(spec.options)}.")
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name}: '{spec.label}' must be a number.") from None
            if spec.kind == "int":
                if not number.is_integer():
                    raise ValueError(f"{name}: '{spec.label}' must be a whole number.")
                number = int(number)
            if not (np.isfinite(number) and spec.min <= number <= spec.max):
                raise ValueError(f"{name}: '{spec.label}' must be between {spec.min} and {spec.max}.")
            value = number
        clean[key] = value
    return clean


def validate_for_data(name: str, params: dict, y_train: np.ndarray) -> None:
    """Checks that depend on the training data (cross-parameter validation)."""
    n_train = len(y_train)
    if name == KNN and params["n_neighbors"] > n_train:
        raise ValueError(f"{name}: k={params['n_neighbors']} is larger than the training set ({n_train}).")
    if name == VOTING and params["n_neighbors"] > n_train:
        raise ValueError(f"{name}: KNN member k is larger than the training set ({n_train}).")
    if name == STACKING:
        smallest = int(np.bincount(y_train).min())
        if params["cv"] > smallest:
            raise ValueError(
                f"{name}: {params['cv']} CV folds need at least {params['cv']} training samples "
                f"in every class, but the smallest class has {smallest}. Lower the number of folds."
            )


def _logreg(C=1.0, seed=42):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=seed))


def _knn(k=5, weights="uniform"):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=weights))


def build_model(name: str, params: dict | None = None, seed: int = 42):
    """Create an *unfitted* scikit-learn estimator."""
    p = validate_params(name, params)
    if name == LOGREG:
        return _logreg(p["C"], seed)
    if name == TREE:
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=seed)
    if name == KNN:
        return _knn(p["n_neighbors"], p["weights"])
    if name == SVM:
        return make_pipeline(StandardScaler(), SVC(C=p["C"], kernel=p["kernel"], random_state=seed))
    if name == FOREST:
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=seed)
    if name == BAGGING:
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["max_depth"]),
            n_estimators=p["n_estimators"], max_samples=p["max_samples"], random_state=seed,
        )
    if name == ADABOOST:
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["max_depth"]),
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], random_state=seed,
        )
    if name == GRADBOOST:
        return GradientBoostingClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], random_state=seed,
        )
    if name == HISTGB:
        return HistGradientBoostingClassifier(
            max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            early_stopping=False, random_state=seed,
        )
    if name == VOTING:
        return VotingClassifier(
            estimators=[
                ("lr", _logreg(1.0, seed)),
                ("dt", DecisionTreeClassifier(max_depth=p["max_depth"], random_state=seed)),
                ("knn", _knn(p["n_neighbors"])),
            ],
            voting=p["voting"],
        )
    if name == STACKING:
        return StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=p["max_depth"], random_state=seed)),
                ("lr", _logreg(1.0, seed)),
                ("rf", RandomForestClassifier(n_estimators=p["rf_estimators"], random_state=seed)),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
            # Out-of-fold predictions: each base model predicts rows it did NOT train on.
            cv=StratifiedKFold(n_splits=p["cv"], shuffle=True, random_state=seed),
        )
    if name == XGB:
        if not xgboost_available():
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        from xgboost import XGBClassifier  # imported lazily: optional dependency

        return XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            random_state=seed, n_jobs=1, verbosity=0, eval_metric="logloss",
        )
    raise ValueError(f"Unknown algorithm: {name!r}")
