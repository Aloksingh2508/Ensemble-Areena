"""Streamlit helpers shared by all pages (widgets + result renderers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import plots
from .data import DATASET_NAMES, SYNTHETIC, Dataset, SplitData, load_dataset, split_data
from .models import CATEGORY, PARAM_SPECS, available_algorithms
from .training import (
    feature_importance, fit_2d_models, is_member_ensemble, member_prediction_table, member_probabilities,
    permutation_importance_df, results_table, staged_scores, stacking_weights, supports_staged,
)
from sklearn.ensemble import StackingClassifier, VotingClassifier

APP_NAME = "Ensemble Learning Visualizer"


# ------------------------------------------------------------ basic helpers
def setup_page(title: str, icon: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def show_plotly(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


def show_df(df, **kwargs) -> None:
    st.dataframe(df, width="stretch", **kwargs)


def explain(text: str, title: str = "ℹ️ What does this mean?") -> None:
    with st.expander(title):
        st.markdown(text)


def get_last_run():
    return st.session_state.get("last_run")


# --------------------------------------------------------------- widgets
def dataset_controls(prefix: str) -> dict:
    """Dataset, test size, seed (+ synthetic options). Returns a plain config dict."""
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Dataset", DATASET_NAMES, index=2, key=f"{prefix}_dataset")
    test_size = c2.slider("Test-set size", 0.10, 0.50, 0.25, 0.05, key=f"{prefix}_test_size",
                          help="Share of the data held back to evaluate the models.")
    seed = int(c3.number_input("Random seed", 0, 9999, 42, 1, key=f"{prefix}_seed",
                               help="Same seed = same split and same results."))
    synth = {}
    if name == SYNTHETIC:
        with st.expander("Synthetic data settings", expanded=True):
            a, b, c, d, e = st.columns(5)
            synth = dict(
                n_samples=a.slider("Samples", 100, 3000, 500, 100, key=f"{prefix}_n"),
                n_features=b.slider("Features", 2, 20, 2, 1, key=f"{prefix}_nf",
                                    help="Use 2 features to see decision boundaries directly."),
                n_classes=c.selectbox("Classes", [2, 3], key=f"{prefix}_nc"),
                class_sep=d.slider("Class separation", 0.3, 3.0, 1.0, 0.1, key=f"{prefix}_sep",
                                   help="Lower = classes overlap more = harder problem."),
                label_noise=e.slider("Label noise", 0.0, 0.3, 0.03, 0.01, key=f"{prefix}_noise",
                                     help="Fraction of labels flipped at random."),
            )
    return {"dataset": name, "test_size": float(test_size), "seed": seed, **synth}


def build_data(cfg: dict) -> tuple[Dataset, SplitData]:
    ds = load_dataset(cfg["dataset"], seed=cfg["seed"], **{k: v for k, v in cfg.items() if k in
                      ("n_samples", "n_features", "n_classes", "class_sep", "label_noise")})
    return ds, split_data(ds, cfg["test_size"], cfg["seed"])


def param_widgets(name: str, prefix: str) -> dict:
    """Build one widget per ParamSpec and return the chosen values."""
    params = {}
    for s in PARAM_SPECS[name]:
        key = f"{prefix}_{name}_{s.key}"
        if s.kind == "int":
            params[s.key] = st.slider(s.label, int(s.min), int(s.max), int(s.default), int(s.step or 1), key=key, help=s.help or None)
        elif s.kind == "float":
            params[s.key] = st.slider(s.label, float(s.min), float(s.max), float(s.default), float(s.step), format="%.2f", key=key, help=s.help or None)
        else:
            params[s.key] = st.selectbox(s.label, list(s.options), index=list(s.options).index(s.default), key=key, help=s.help or None)
    return params


def algorithm_selector(prefix: str, default: list[str]) -> dict:
    names = st.multiselect("Choose one or more algorithms", available_algorithms(), default=default, key=f"{prefix}_algos")
    selected = {}
    cols = st.columns(2) if len(names) > 1 else [st.container()]
    for i, n in enumerate(names):
        with cols[i % len(cols)]:
            with st.expander(f"⚙️ {n}  ·  {CATEGORY[n]}"):
                selected[n] = param_widgets(n, prefix)
    return selected


# ------------------------------------------------------------- renderers
def highlight_wrong(df: pd.DataFrame, truth: str = "True label"):
    def per_col(col):
        if col.name == truth:
            return [""] * len(col)
        return ["background-color: rgba(228,87,86,0.35)" if v != t else "background-color: rgba(84,162,75,0.25)"
                for v, t in zip(col, df[truth])]
    return df.style.apply(per_col, axis=0)


def render_dataset(run: dict, kp: str = "ds") -> None:
    ds = run["dataset"]
    if ds.n_features == 2:
        fig = plots.scatter_2d(ds.X, ds.y, ds.class_names, ds.feature_names[0], ds.feature_names[1], f"{ds.name} - all samples")
    else:
        mode = st.radio("View", ["PCA (2 components)", "Pick two features"], horizontal=True, key=f"{kp}_view")
        if mode.startswith("PCA"):
            X2, ev = plots.pca_project(ds.X)
            fig = plots.scatter_2d(X2, ds.y, ds.class_names, f"PC1 ({ev[0]:.0%} of variance)", f"PC2 ({ev[1]:.0%})", f"{ds.name} - PCA projection")
        else:
            a, b = st.columns(2)
            f1 = a.selectbox("X axis", ds.feature_names, 0, key=f"{kp}_f1")
            f2 = b.selectbox("Y axis", ds.feature_names, 1, key=f"{kp}_f2")
            fig = plots.scatter_2d(ds.X[:, [ds.feature_names.index(f1), ds.feature_names.index(f2)]], ds.y, ds.class_names, f1, f2, ds.name)
    show_plotly(fig, key=f"{kp}_scatter")
    explain("Each dot is one sample, coloured by its true class. The more the colours overlap, the harder the "
            "classification problem. PCA squeezes many features into 2 axes just for drawing - nothing is trained on it.")


def render_scores(run: dict, kp: str = "sc") -> None:
    df = results_table(run["results"])
    show_df(df.style.format({m: "{:.3f}" for m in plots.METRICS} | {"Train time (s)": "{:.4f}"}).highlight_max(subset=plots.METRICS, color="rgba(84,162,75,0.35)"))
    show_plotly(plots.metric_bars(df), key=f"{kp}_bars")
    explain("**Accuracy** = share of correct predictions. **Precision** = when the model says 'class X', how often is it right? "
            "**Recall** = of all real class-X samples, how many did it find? **F1** balances precision and recall. "
            "For several classes these are averaged (weighted by class size). A tiny difference (e.g. 0.01) on a small "
            "test set may just be luck - change the seed to check.")
    show_plotly(plots.time_bars(df), key=f"{kp}_time")
    explain("Ensembles train many models, so they are usually slower than a single model. Stacking is slowest because it "
            "also runs cross-validation. Accuracy is not the only thing that matters - speed and simplicity count too.")


def render_overview(run: dict, kp: str = "ov") -> None:
    df = results_table(run["results"])
    best, fastest = df.iloc[0], df.sort_values("Train time (s)").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best accuracy", f"{best['Accuracy']:.3f}", best["Algorithm"], delta_color="off")
    c2.metric("Best F1-score", f"{df['F1-score'].max():.3f}", df.sort_values("F1-score").iloc[-1]["Algorithm"], delta_color="off")
    c3.metric("Fastest to train", f"{fastest['Train time (s)'] * 1000:.0f} ms", fastest["Algorithm"], delta_color="off")
    c4.metric("Models trained", len(df), f"{len(run['split'].y_train)} train / {len(run['split'].y_test)} test", delta_color="off")
    st.subheader("1 · Dataset")
    render_dataset(run, kp)
    st.subheader("2 · Scores")
    render_scores(run, kp)


def render_confusions(run: dict, kp: str = "cm") -> None:
    results, names = run["results"], run["split"].class_names
    cols = st.columns(2)
    for i, r in enumerate(results):
        with cols[i % 2]:
            show_plotly(plots.confusion_heatmap(r.confusion, names, f"{r.name} (acc {r.metrics['Accuracy']:.2f})"), key=f"{kp}_{i}")
    explain("Rows = the **true** class, columns = what the model **predicted**. The diagonal (top-left to bottom-right) "
            "counts correct predictions. Numbers off the diagonal are mistakes - they show *which* classes get confused "
            "with each other, which accuracy alone cannot tell you.")


def render_boundaries(run: dict, kp: str = "bd") -> None:
    ds, split, results = run["dataset"], run["split"], run["results"]
    idx = (0, 1)
    f1, f2 = ds.feature_names[0], ds.feature_names[1]
    if ds.n_features > 2:
        a, b = st.columns(2)
        f1 = a.selectbox("X axis feature", ds.feature_names, 0, key=f"{kp}_f1")
        f2 = b.selectbox("Y axis feature", ds.feature_names, 1, key=f"{kp}_f2")
        if f1 == f2:
            st.warning("Please choose two different features.")
            return
        idx = (ds.feature_names.index(f1), ds.feature_names.index(f2))
        st.info("A boundary can only be drawn in 2-D, so every model is **re-trained on just these two features** here. "
                "Its accuracy will differ from the full-feature scores. Tip: pick 2 features in the Synthetic dataset to skip this.")
    cache = run.setdefault("boundary_cache", {})
    if idx not in cache:
        with st.spinner("Training 2-feature versions of the models..."):
            cache[idx] = fit_2d_models({r.name: r.params for r in results}, split, idx, run["seed"])
    if not cache[idx]:
        st.warning("No model could be trained on these two features.")
        return
    fig = plots.decision_boundaries(cache[idx], split.X_train[:, idx], split.y_train, split.X_test[:, idx], split.y_test, f1, f2)
    st.pyplot(fig)
    explain("Coloured regions show the class the model would predict for *any* point. **Circles** are training samples, "
            "**triangles** are test samples. A single deep tree draws jagged, blocky regions (it memorises); bagged and boosted "
            "ensembles usually give smoother boundaries; linear models draw straight lines. Triangles sitting in the wrong colour are test errors.")


def render_predictions(run: dict, kp: str = "pr", part: str = "both") -> None:
    if part in ("table", "both"):
        _prediction_table(run, kp)
    if part in ("members", "both"):
        _ensemble_members(run, kp)


def _prediction_table(run: dict, kp: str) -> None:
    split, results = run["split"], run["results"]
    names = np.array(split.class_names)
    n = st.slider("How many test samples to show", 5, min(30, len(split.y_test)), min(10, len(split.y_test)), key=f"{kp}_n")
    table = {"True label": names[split.y_test[:n]]}
    for r in results:
        table[r.name] = names[r.y_pred[:n]]
    df = pd.DataFrame(table)
    df.index = [f"Test sample {i}" for i in range(n)]
    show_df(highlight_wrong(df))
    explain("Green = the model got that sample right, red = wrong. Compare columns: samples where models **disagree** are "
            "exactly where combining them can help (or hurt).")



def _ensemble_members(run: dict, kp: str) -> None:
    split, results = run["split"], run["results"]
    st.subheader("Inside an ensemble: members vs. final answer")
    n = st.slider("Test samples to show", 3, min(20, len(split.y_test)), min(8, len(split.y_test)), key=f"{kp}_nsamp")
    ens = [r for r in results if is_member_ensemble(r.model)]
    if not ens:
        st.info("Select a Voting, Stacking, Bagging or Random Forest model to see its individual members here.")
        return
    r = next(x for x in ens if x.name == st.selectbox("Ensemble", [x.name for x in ens], key=f"{kp}_ens"))
    n_members = 5
    if r.category == "Bagging":
        n_members = st.slider("Trees to display", 3, 10, 5, key=f"{kp}_nm")
    members = member_prediction_table(r.model, split.X_test, split.y_test, split.class_names, n, n_members)
    show_df(highlight_wrong(members))
    explain("Each column is one member's own prediction; **ENSEMBLE (final)** is the combined answer. "
            "Notice rows where some members are red but the final answer is green: the ensemble out-voted the mistake. "
            "The opposite can also happen.")
    if isinstance(r.model, VotingClassifier):
        s = st.slider("Sample to inspect", 0, len(split.y_test) - 1, 0, key=f"{kp}_s")
        probs = member_probabilities(r.model, split.X_test[s], split.class_names)
        show_plotly(plots.member_probability_bars(probs), key=f"{kp}_vp")
        mode = r.params["voting"]
        explain("**Hard** voting only looks at each member's top class and takes the majority. **Soft** voting averages the bars "
                "(the last group) and picks the tallest. This model uses **%s** voting." % mode)
    if isinstance(r.model, StackingClassifier):
        show_plotly(plots.weights_bars(stacking_weights(r.model, len(split.class_names))), key=f"{kp}_sw")
        explain("The meta-model is a logistic regression trained on the base models' out-of-fold probabilities. A taller bar means the "
                "meta-model relies more on that base model. This is what *stacking learns* and *voting does not*.")


def render_importance(run: dict, kp: str = "fi") -> None:
    split, results = run["split"], run["results"]
    r = next(x for x in results if x.name == st.selectbox("Model", [x.name for x in results], key=f"{kp}_m"))
    method = st.radio("Method", ["Built-in (tree impurity)", "Permutation (test set)"], horizontal=True, key=f"{kp}_meth")
    if method.startswith("Built-in"):
        df = feature_importance(r.model, split.feature_names)
        if df is None:
            st.info(f"{r.name} has no built-in feature importance. Try the **Permutation** method - it works for any model.")
            return
        title = f"Built-in importance - {r.name}"
    else:
        cache = run.setdefault("perm_cache", {})
        if r.name not in cache:
            with st.spinner("Shuffling features and re-scoring..."):
                cache[r.name] = permutation_importance_df(r.model, split, run["seed"])
        df, title = cache[r.name], f"Permutation importance - {r.name}"
    show_plotly(plots.importance_bars(df, title), key=f"{kp}_bar")
    explain("A longer bar means the model leans more on that feature. **Built-in** importance counts how much a feature helps tree "
            "splits (measured on training data). **Permutation** importance shuffles one feature on the test set and measures how much "
            "accuracy drops. Caution: importance is *not* causation, and correlated features split the credit between them.")


def render_boosting(run: dict, kp: str = "bo") -> None:
    split, results = run["split"], run["results"]
    boosters = [r for r in results if supports_staged(r.model)]
    if not boosters:
        st.info("Add **AdaBoost**, **Gradient Boosting** or **HistGradientBoosting** to see how accuracy changes after every boosting iteration.")
        return
    cache = run.setdefault("staged_cache", {})
    curves = {}
    for r in boosters:
        if r.name not in cache:
            cache[r.name] = staged_scores(r.model, split)
        curves[r.name] = cache[r.name]
    show_plotly(plots.staged_lines(curves), key=f"{kp}_lines")
    rows = [{"Algorithm": n, "Best iteration (test)": int(c.loc[c["Test accuracy"].idxmax(), "Iteration"]),
             "Best test accuracy": c["Test accuracy"].max(), "Final test accuracy": c["Test accuracy"].iloc[-1],
             "Final train accuracy": c["Train accuracy"].iloc[-1]} for n, c in curves.items()]
    show_df(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["Best test accuracy", "Final test accuracy", "Final train accuracy"]}))
    explain("Boosting adds one small model at a time. The curve shows accuracy using only the first *k* models. Early on, accuracy usually "
            "climbs quickly as new models fix earlier mistakes. **Solid** lines = test data, **dotted** = training data. If the dotted line "
            "keeps rising while the solid line flattens or falls, the model is starting to overfit - fewer iterations or a smaller learning rate may help.")
