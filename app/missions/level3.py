"""Level 3 - Boosting Challenge: AdaBoost beats a single stump by focusing on mistakes."""
import streamlit as st

from src import plots
from src.models import ADABOOST, TREE
from src.training import staged_scores, train_and_evaluate
from app.game import state
from app.missions.common import arena_data, check_question, finish, flags, objective
from src.ui import show_plotly

MID, MIN_EST = 3, 20


def render() -> None:
    ds, split = arena_data()
    st.markdown("**Boosting** adds weak models one at a time. After each round, the samples that were **misclassified get more weight**, "
                "so the next model focuses on the mistakes. **Your task:** beat a single *stump* (a depth-1 tree) with AdaBoost using "
                f"at least **{MIN_EST} estimators**.")
    a, b = st.columns(2)
    n = a.slider("Number of estimators", 1, 200, 5, key="m3_n")
    lr = b.slider("Learning rate", 0.05, 2.0, 1.0, 0.05, key="m3_lr")
    if st.button("🚀 Train AdaBoost", type="primary", key="m3_train"):
        ada = train_and_evaluate(ADABOOST, {"n_estimators": n, "learning_rate": lr, "max_depth": 1}, split, 7)
        stump = train_and_evaluate(TREE, {"max_depth": 1}, split, 7)
        st.session_state["m3"] = dict(ada=ada, stump=stump, curve=staged_scores(ada.model, split), n=n)
        state.count_training()
        if n >= MIN_EST and ada.metrics["Accuracy"] > stump.metrics["Accuracy"]:
            flags()["m3"] = True
    r = st.session_state.get("m3")
    if r:
        c = st.columns(3)
        c[0].metric("Single stump", f"{r['stump'].metrics['Accuracy']:.3f}")
        c[1].metric(f"AdaBoost ({r['n']} models)", f"{r['ada'].metrics['Accuracy']:.3f}",
                    f"{r['ada'].metrics['Accuracy'] - r['stump'].metrics['Accuracy']:+.3f}")
        c[2].metric("F1-score", f"{r['ada'].metrics['F1-score']:.3f}")
        show_plotly(plots.staged_lines({"AdaBoost": r["curve"]}), key="m3_curve")
        last = len(r["curve"])
        st.pyplot(plots.boosting_snapshots(r["ada"].model, split.X_train, split.y_train, sorted({1, min(3, last), min(10, last), last})))
        st.caption("Watch the boundary: with 1 stump it is a straight cut; each added model bends it to fix earlier mistakes.")
        if r["n"] >= MIN_EST and not flags().get("m3"):
            st.info("AdaBoost did not beat the stump this time - try a different learning rate.")
    objective(flags().get("m3", False), f"AdaBoost with ≥ {MIN_EST} estimators scores higher than the single stump")
    check_question(MID, "In boosting, what do later models focus on?", ["Random samples", "The easiest samples",
                   "The samples earlier models got wrong", "The test set"], 2, "Mistakes get more weight (AdaBoost) or are fitted directly (Gradient Boosting).")
    finish(MID, flags().get("m3", False))
