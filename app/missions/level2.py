"""Level 2 - Random Forest Squad: bootstrap sampling, tree votes, accuracy vs. number of trees."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import plots
from src.models import FOREST, TREE
from src.training import member_prediction_table, train_and_evaluate
from app.game import state
from app.missions.common import arena_data, check_question, finish, flags, objective
from src.ui import show_df, show_plotly

MID, MIN_TREES = 2, 30


def _bootstrap_html(n: int = 20, trees: int = 3, seed: int = 0) -> str:
    out = ""
    for t in range(trees):
        counts = np.bincount(np.random.RandomState(seed + t).choice(n, n, replace=True), minlength=n)
        dots = "".join(f'<span class="dot {"on" if c else "off"}" title="row {i}: used {c}x">{c if c > 1 else ""}</span>' for i, c in enumerate(counts))
        out += f'<div class="boot-row"><b>Tree {t + 1}</b> &nbsp; {dots}</div>'
    return out


def render() -> None:
    ds, split = arena_data()
    st.markdown("A **Random Forest** is a squad of decision trees. Each tree trains on a different **bootstrap sample** and the squad **votes**. "
                "Averaging many unstable trees reduces **variance**.")
    st.subheader("1 · Bootstrap sampling")
    st.markdown("Each row of dots is one tree's training set of 20 rows. **Filled** = row used (number = used more than once), **faded** = left out (out-of-bag).",)
    seed = st.slider("Draw", 0, 40, 0, key="m2_boot")
    st.markdown(_bootstrap_html(20, 3, seed * 3), unsafe_allow_html=True)

    st.subheader("2 · Train your squad")
    n_trees = st.slider("Number of trees", 1, 100, 10, key="m2_trees")
    if st.button("🌲 Train Forest", type="primary", key="m2_train"):
        forest = train_and_evaluate(FOREST, {"n_estimators": n_trees, "max_depth": 10}, split, 7)
        tree = train_and_evaluate(TREE, {"max_depth": 20}, split, 7)
        sweep = [{"Trees": n, "Forest accuracy": train_and_evaluate(FOREST, {"n_estimators": n, "max_depth": 10}, split, 7).metrics["Accuracy"]}
                 for n in (1, 3, 5, 10, 20, 40, 70, 100)]
        st.session_state["m2"] = dict(forest=forest, tree=tree, sweep=pd.DataFrame(sweep), n=n_trees)
        state.count_training()
        if n_trees >= MIN_TREES:
            flags()["m2"] = True
    r = st.session_state.get("m2")
    if r:
        f, t = r["forest"], r["tree"]
        c = st.columns(5)
        for col, (k, v) in zip(c[:4], f.metrics.items()):
            col.metric(k, f"{v:.2f}")
        c[4].metric("Single deep tree", f"{t.metrics['Accuracy']:.2f}")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Individual predictions → majority vote** ({r['n']} trees)")
            s = st.slider("Test sample", 0, len(split.y_test) - 1, 0, key="m2_s")
            x = split.X_test[s:s + 1]
            tbl = member_prediction_table(f.model, x, split.y_test[s:s + 1], ds.class_names, 1, 3)
            show_df(tbl)
            votes = np.bincount(np.array([int(e.predict(x)[0]) for e in f.model.estimators_]), minlength=2)
            st.caption(f"All {r['n']} trees voted: Class 0 → **{votes[0]}**, Class 1 → **{votes[1]}**  ⇒ forest answer: **{ds.class_names[int(f.model.predict(x)[0])]}**")
        with right:
            fig = px.line(r["sweep"], x="Trees", y="Forest accuracy", markers=True, title="Accuracy vs. number of trees")
            fig.add_hline(y=t.metrics["Accuracy"], line_dash="dash", annotation_text="single deep tree")
            fig.update_layout(height=330, margin=dict(t=50, b=10))
            show_plotly(fig, key="m2_sweep")
        cols2 = (0, 1)
        st.pyplot(plots.decision_boundaries({f"Random Forest ({r['n']} trees)": f.model, "Single deep tree": t.model},
                                            split.X_train, split.y_train, split.X_test, split.y_test, "x1", "x2", 2))
        st.caption("Compare the jagged single tree with the smoother forest. More trees usually means smoother boundaries and steadier accuracy.")
    objective(flags().get("m2", False), f"train a forest with at least {MIN_TREES} trees")
    check_question(MID, "Why does averaging many trees reduce variance?", ["Each tree gets deeper", "Individual trees' random mistakes tend to cancel out",
                   "The trees see the test set", "It makes training faster"], 1,
                   "Different bootstrap samples make different mistakes; the majority vote smooths them out.")
    finish(MID, flags().get("m2", False))
