"""Plotly and Matplotlib figure builders (return figures; never call Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]


def _colors(class_names):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(class_names)}


# ------------------------------------------------------------- dataset
def scatter_2d(X2, y, class_names, xlabel: str, ylabel: str, title: str = "") -> go.Figure:
    df = pd.DataFrame({xlabel: X2[:, 0], ylabel: X2[:, 1], "Class": np.array(class_names)[np.asarray(y).astype(int)]})
    fig = px.scatter(df, x=xlabel, y=ylabel, color="Class", color_discrete_map=_colors(class_names),
                     category_orders={"Class": list(class_names)}, title=title, opacity=0.8)
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def pca_project(X) -> tuple[np.ndarray, list[float]]:
    """2-D PCA used ONLY for plotting the raw dataset (nothing is trained on it)."""
    pca = PCA(n_components=2).fit(X)
    return pca.transform(X), list(pca.explained_variance_ratio_)


# ------------------------------------------------------------- metrics
def metric_bars(df: pd.DataFrame) -> go.Figure:
    long = df.melt(id_vars="Algorithm", value_vars=METRICS, var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Algorithm", y="Score", color="Metric", barmode="group", title="Test-set performance")
    lo = max(0.0, float(long["Score"].min()) - 0.1)
    fig.update_yaxes(range=[lo, 1.0])
    fig.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
    return fig


def time_bars(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("Train time (s)")
    fig = px.bar(d, x="Train time (s)", y="Algorithm", orientation="h", title="Training time (seconds)",
                 color="Type")
    fig.update_layout(height=max(300, 45 * len(d) + 120), margin=dict(t=50, b=10, l=10, r=10))
    return fig


def accuracy_vs_time(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="Train time (s)", y="Accuracy", color="Type", text="Algorithm",
                     title="Accuracy vs. training time (top-left is best)")
    fig.update_traces(textposition="top center", marker=dict(size=13))
    fig.update_layout(height=450, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def radar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in df.iterrows():
        vals = [row[m] for m in METRICS]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=METRICS + METRICS[:1], name=row["Algorithm"], fill="toself", opacity=0.45))
    lo = max(0.0, float(df[METRICS].min().min()) - 0.05)
    fig.update_layout(polar=dict(radialaxis=dict(range=[lo, 1.0])), height=450, margin=dict(t=30, b=10))
    return fig


def confusion_heatmap(cm, class_names, title: str) -> go.Figure:
    fig = px.imshow(cm, x=list(class_names), y=list(class_names), text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"), title=title, aspect="auto")
    fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10), coloraxis_showscale=False)
    return fig


def agreement_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(df.values, x=list(df.columns), y=list(df.index), text_auto=".2f", zmin=df.values.min(), zmax=1.0,
                    color_continuous_scale="Viridis", title="How often two models give the same answer")
    fig.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def importance_bars(df: pd.DataFrame, title: str, top: int = 15) -> go.Figure:
    d = df.head(top).iloc[::-1]
    fig = px.bar(d, x="Importance", y="Feature", orientation="h", title=title)
    fig.update_layout(height=max(280, 28 * len(d) + 120), margin=dict(t=50, b=10, l=10, r=10))
    return fig


def staged_lines(curves: dict) -> go.Figure:
    """``curves``: algorithm -> DataFrame from ``staged_scores``."""
    fig = go.Figure()
    for i, (name, df) in enumerate(curves.items()):
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=df["Iteration"], y=df["Test accuracy"], name=f"{name} - test", line=dict(color=c)))
        fig.add_trace(go.Scatter(x=df["Iteration"], y=df["Train accuracy"], name=f"{name} - train", line=dict(color=c, dash="dot")))
    fig.update_layout(title="Accuracy after each boosting iteration", xaxis_title="Boosting iteration (models added so far)",
                      yaxis_title="Accuracy", height=430, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def member_probability_bars(df: pd.DataFrame) -> go.Figure:
    long = df.reset_index(names="Member").melt(id_vars="Member", var_name="Class", value_name="Probability")
    fig = px.bar(long, x="Member", y="Probability", color="Class", barmode="group", title="Class probabilities for the selected sample")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def weights_bars(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="Base model", y="Meta-model weight", title="How much the meta-model trusts each base model")
    fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def variance_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="Model", y="Test accuracy", points="all", color="Model", title="Test accuracy over 15 different training subsets")
    fig.update_layout(height=400, showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def vote_curve(ns, accs, p: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=ns, y=accs, mode="lines+markers", name="Majority vote"))
    fig.add_hline(y=p, line_dash="dash", annotation_text=f"single model ({p:.2f})")
    fig.update_layout(title="Majority-vote accuracy vs. number of independent models", xaxis_title="Number of models (odd)",
                      yaxis_title="Ensemble accuracy", height=380, margin=dict(t=50, b=10, l=10, r=10))
    return fig


# ---------------------------------------------- Matplotlib: boundaries
def _grid(X2, resolution=140, pad=0.5):
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _draw_points(ax, X_train, y_train, X_test, y_test, cmap):
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, s=16, edgecolors="white", linewidths=0.4)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, s=34, marker="^", edgecolors="black", linewidths=0.7)


def decision_boundaries(models: dict, X_train2, y_train, X_test2, y_test, xlabel, ylabel, max_cols: int = 3) -> Figure:
    """One panel per model. Circles = training points, triangles = test points."""
    cmap = ListedColormap(PALETTE)
    n = len(models)
    cols = min(max_cols, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig = Figure(figsize=(4.6 * cols, 3.9 * rows))
    axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
    X_all = np.vstack([X_train2, X_test2])
    xx, yy, grid = _grid(X_all)
    for ax, (name, model) in zip(axes, models.items()):
        zz = np.asarray(model.predict(grid)).reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, alpha=0.30, levels=np.arange(-0.5, len(PALETTE), 1))
        _draw_points(ax, X_train2, y_train, X_test2, y_test, cmap)
        ax.set_title(f"{name}\ntest acc = {model.score(X_test2, y_test):.2f}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def boosting_snapshots(model, X_train2, y_train, stages: list[int], xlabel="x1", ylabel="x2") -> Figure:
    """Decision boundary of a fitted boosting model after only the first k iterations."""
    cmap = ListedColormap(PALETTE)
    xx, yy, grid = _grid(X_train2)
    wanted = set(stages)
    snaps = {i: np.asarray(p).reshape(xx.shape) for i, p in enumerate(model.staged_predict(grid), start=1) if i in wanted}
    stages = [s for s in stages if s in snaps]
    fig = Figure(figsize=(4.0 * len(stages), 3.6))
    axes = np.atleast_1d(fig.subplots(1, len(stages), squeeze=False)).ravel()
    for ax, s in zip(axes, stages):
        ax.contourf(xx, yy, snaps[s], cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, alpha=0.30, levels=np.arange(-0.5, len(PALETTE), 1))
        ax.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, s=14, edgecolors="white", linewidths=0.4)
        ax.set_title(f"After {s} model{'s' if s > 1 else ''}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(ylabel, fontsize=8)
    fig.tight_layout()
    return fig
