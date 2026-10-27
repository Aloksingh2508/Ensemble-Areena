"""Unit tests for data handling, model construction, training and evaluation."""
import numpy as np
import pytest
from sklearn.ensemble import StackingClassifier, VotingClassifier

from src.data import BREAST_CANCER, IRIS, SYNTHETIC, SplitData, load_dataset, make_demo_data, split_data
from src.models import (ADABOOST, FOREST, KNN, LOGREG, STACKING, TREE, VOTING, available_algorithms,
                        build_model, get_default_params, validate_params)
from src.training import (bootstrap_demo, compute_metrics, cross_validate_model, feature_importance,
                          fit_2d_models, majority_vote_accuracy, member_prediction_table, member_probabilities,
                          oob_vs_test, pairwise_agreement, results_table, run_experiment, stacking_weights,
                          staged_scores, supports_staged, train_and_evaluate, variance_experiment)


@pytest.fixture(scope="module")
def split():
    return split_data(load_dataset(SYNTHETIC, n_samples=300, n_features=4, seed=0), 0.25, 0)


# ------------------------------------------------------------------ data
@pytest.mark.parametrize("name,shape,k", [(IRIS, (150, 4), 3), (BREAST_CANCER, (569, 30), 2)])
def test_builtin_datasets(name, shape, k):
    ds = load_dataset(name)
    assert ds.X.shape == shape and ds.n_classes == k


@pytest.mark.parametrize("kw", [dict(n_samples=50), dict(n_features=1), dict(n_classes=5),
                                dict(class_sep=9), dict(label_noise=0.9)])
def test_synthetic_validation(kw):
    with pytest.raises(ValueError):
        load_dataset(SYNTHETIC, **kw)


@pytest.mark.parametrize("n_classes,n_features", [(2, 2), (3, 2), (3, 3), (2, 20), (3, 20)])
def test_synthetic_combinations_work(n_classes, n_features):
    ds = load_dataset(SYNTHETIC, n_samples=200, n_features=n_features, n_classes=n_classes)
    assert ds.X.shape == (200, n_features) and len(np.unique(ds.y)) == n_classes


def test_split_is_stratified_disjoint_and_reproducible(split):
    ds = load_dataset(SYNTHETIC, n_samples=300, n_features=4, seed=0)
    again = split_data(ds, 0.25, 0)
    assert np.array_equal(split.X_train, again.X_train)
    assert len(split.y_train) + len(split.y_test) == 300 and len(split.y_test) == 75
    train_rows = {tuple(r) for r in split.X_train}
    assert not any(tuple(r) in train_rows for r in split.X_test)  # continuous data -> no shared rows
    full = np.bincount(ds.y) / 300
    assert np.allclose(np.bincount(split.y_test) / 75, full, atol=0.03)  # stratified


@pytest.mark.parametrize("ts,seed", [(0.05, 1), (0.9, 1), (0.2, -1), (0.2, 1.5)])
def test_split_rejects_bad_input(ts, seed):
    with pytest.raises(ValueError):
        split_data(load_dataset(IRIS), ts, seed)


def test_demo_data():
    assert make_demo_data(200).X.shape == (200, 2)


# ---------------------------------------------------------------- models
def test_defaults_valid_for_every_algorithm():
    for name in available_algorithms():
        assert validate_params(name, get_default_params(name))


@pytest.mark.parametrize("name,bad", [(TREE, {"max_depth": 0}), (TREE, {"max_depth": 2.5}), (TREE, {"max_depth": "x"}),
                                      (FOREST, {"n_estimators": 10**6}), (VOTING, {"voting": "median"}),
                                      (TREE, {"nope": 1}), ("Nonexistent", {})])
def test_invalid_params_rejected(name, bad):
    with pytest.raises(ValueError):
        build_model(name, bad)


def test_stacking_cv_larger_than_smallest_class_is_rejected():
    rng = np.random.RandomState(0)
    X = rng.randn(63, 2)
    y = np.array([0] * 30 + [1] * 30 + [2] * 3)  # class 2 has only 3 samples
    sp = SplitData(X, X[:10], y, y[:10], ["a", "b"], ["0", "1", "2"])
    with pytest.raises(ValueError, match="folds"):
        train_and_evaluate(STACKING, {"cv": 5}, sp)


def test_run_experiment_reports_errors_without_crashing(split):
    results, errors = run_experiment({TREE: None, STACKING: {"cv": 99}, FOREST: {"n_estimators": 10}}, split)
    assert {r.name for r in results} == {TREE, FOREST}
    assert STACKING in errors


# ---------------------------------------------------------- training
@pytest.mark.parametrize("name", available_algorithms())
def test_every_algorithm_trains_and_scores(name, split):
    r = train_and_evaluate(name, None, split, seed=1)
    assert 0.5 < r.metrics["Accuracy"] <= 1.0
    assert r.confusion.sum() == len(split.y_test)
    assert r.train_time >= 0 and len(r.y_pred) == len(split.y_test)


def test_training_is_deterministic(split):
    a = train_and_evaluate(FOREST, {"n_estimators": 20}, split, 3)
    b = train_and_evaluate(FOREST, {"n_estimators": 20}, split, 3)
    assert np.array_equal(a.y_pred, b.y_pred)


def test_metrics_known_values():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["Accuracy"] == 0.75
    assert m["Recall"] == pytest.approx(0.75)
    assert compute_metrics([0, 1], [1, 0])["F1-score"] == 0.0  # zero_division handled


def test_multiclass_metrics_and_confusion():
    sp = split_data(load_dataset(IRIS), 0.3, 0)
    r = train_and_evaluate(FOREST, {"n_estimators": 20}, sp)
    assert r.confusion.shape == (3, 3)


def test_no_leakage_scaler_fitted_on_training_data_only(split):
    model = build_model(LOGREG, None, 0).fit(split.X_train, split.y_train)
    scaler = model.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, split.X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([split.X_train, split.X_test]).mean(axis=0))


def test_results_table_sorted_by_accuracy(split):
    results, _ = run_experiment({TREE: None, FOREST: None, KNN: None}, split)
    df = results_table(results)
    assert list(df["Accuracy"]) == sorted(df["Accuracy"], reverse=True)


def test_cross_validation_and_agreement(split):
    mean, std = cross_validate_model(TREE, None, split, 0, 3)
    assert 0.5 < mean <= 1 and std >= 0
    results, _ = run_experiment({TREE: None, FOREST: None}, split)
    agree = pairwise_agreement(results)
    assert np.allclose(np.diag(agree.values), 1.0) and agree.shape == (2, 2)


# ------------------------------------------------- ensemble introspection
def test_boosting_staged_scores(split):
    r = train_and_evaluate(ADABOOST, {"n_estimators": 20}, split)
    assert supports_staged(r.model)
    df = staged_scores(r.model, split)
    assert 1 <= len(df) <= 20 and df["Test accuracy"].iloc[-1] == pytest.approx(r.metrics["Accuracy"])
    assert staged_scores(train_and_evaluate(TREE, None, split).model, split) is None


def test_voting_members_and_probabilities(split):
    r = train_and_evaluate(VOTING, {"voting": "soft"}, split)
    assert isinstance(r.model, VotingClassifier)
    tbl = member_prediction_table(r.model, split.X_test, split.y_test, split.class_names, 5)
    assert list(tbl.columns) == ["True label", LOGREG, TREE, KNN, "ENSEMBLE (final)"]
    # ensemble column must equal the real model's predictions
    expected = np.array(split.class_names)[r.model.predict(split.X_test[:5])]
    assert list(tbl["ENSEMBLE (final)"]) == list(expected)
    probs = member_probabilities(r.model, split.X_test[0], split.class_names)
    assert np.allclose(probs.sum(axis=1), 1.0) and probs.shape == (4, 2)


def test_soft_vote_equals_average_of_member_probabilities(split):
    r = train_and_evaluate(VOTING, {"voting": "soft"}, split)
    probs = member_probabilities(r.model, split.X_test[3], split.class_names)
    assert int(np.argmax(probs.loc["Average (soft vote)"].values)) == int(r.model.predict(split.X_test[3:4])[0])


@pytest.mark.parametrize("name", ["Random Forest", "Bagging Classifier"])
def test_tree_ensemble_member_table(name, split):
    r = train_and_evaluate(name, {"n_estimators": 10}, split)
    tbl = member_prediction_table(r.model, split.X_test, split.y_test, split.class_names, 4, 3)
    assert tbl.shape == (4, 5)  # truth + 3 trees + final


def test_stacking_weights(split):
    r = train_and_evaluate(STACKING, {"cv": 3, "rf_estimators": 10}, split)
    assert isinstance(r.model, StackingClassifier)
    w = stacking_weights(r.model, 2)
    assert len(w) == 3 and (w["Meta-model weight"] >= 0).all()
    ds3 = split_data(load_dataset(IRIS), 0.3, 0)
    r3 = train_and_evaluate(STACKING, {"cv": 3, "rf_estimators": 10}, ds3)
    assert len(stacking_weights(r3.model, 3)) == 3


def test_feature_importance(split):
    rf = train_and_evaluate(FOREST, {"n_estimators": 10}, split)
    df = feature_importance(rf.model, split.feature_names)
    assert df["Importance"].sum() == pytest.approx(1.0) and len(df) == 4
    bag = train_and_evaluate("Bagging Classifier", {"n_estimators": 10}, split)
    assert feature_importance(bag.model, split.feature_names) is not None
    assert feature_importance(train_and_evaluate(KNN, None, split).model, split.feature_names) is None


def test_fit_2d_models(split):
    models = fit_2d_models({TREE: None, FOREST: {"n_estimators": 10}}, split, (0, 1))
    assert set(models) == {TREE, FOREST}
    assert models[TREE].predict(split.X_test[:, [0, 1]]).shape == (len(split.y_test),)


# ---------------------------------------------------------- theory demos
def test_majority_vote_accuracy_math():
    assert majority_vote_accuracy(0.7, 1) == pytest.approx(0.7)
    assert majority_vote_accuracy(0.6, 3) == pytest.approx(0.648)
    assert majority_vote_accuracy(0.5, 11) == pytest.approx(0.5)
    assert majority_vote_accuracy(0.6, 51) > majority_vote_accuracy(0.6, 3)
    with pytest.raises(ValueError):
        majority_vote_accuracy(0.6, 4)


def test_bootstrap_demo_and_experiments(split):
    d = bootstrap_demo(100, 1)
    assert len(d["draw"]) == 100 and d["unique"] + len(d["out_of_bag"]) == 100
    assert 55 <= d["unique"] <= 72  # ~63.2%
    v = variance_experiment(split, n_runs=3, n_estimators=5)
    assert len(v) == 6
    o = oob_vs_test(split, 40)
    assert 0 < o["oob_accuracy"] <= 1 and 0 < o["test_accuracy"] <= 1
