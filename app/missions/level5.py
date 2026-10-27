"""Level 5 - Stacking Mastership: a meta-model learns how to combine base models."""
import pandas as pd
import streamlit as st

from src import plots
from src.models import STACKING
from src.training import stacking_weights, train_and_evaluate
from app.game import state
from app.missions.common import arena_data, check_question, finish, flags, objective
from src.ui import show_plotly

MID = 5


def render() -> None:
    ds, split = arena_data()
    st.markdown("**Stacking** trains several base models (here: Decision Tree, Logistic Regression, Random Forest), collects their "
                "**out-of-fold** predictions, and trains a **meta-model** to combine them. **Objective:** train a stack and compare it with its base models.")
    cv = st.slider("Out-of-fold CV folds", 2, 10, 5, key="m5_cv")
    if st.button("🧠 Train Stack", type="primary", key="m5_train"):
        with st.spinner("Training base models and the stack..."):
            rs = [train_and_evaluate(n, None, split, 7) for n in ("Decision Tree", "Logistic Regression", "Random Forest")]
            rs.append(train_and_evaluate(STACKING, {"cv": cv}, split, 7))
        st.session_state["m5"] = rs
        state.count_training()
        flags()["m5"] = True
    rs = st.session_state.get("m5")
    if rs:
        show_plotly(plots.metric_bars(pd.DataFrame([{"Algorithm": r.name, **r.metrics} for r in rs])), key="m5_bars")
        show_plotly(plots.weights_bars(stacking_weights(rs[-1].model, 2)), key="m5_w")
        st.caption("The second chart shows how much the meta-model leans on each base model. The stack is **not guaranteed** to win - check the first chart!")
    objective(flags().get("m5", False), "train the stacking model")
    check_question(MID, "Why does stacking use out-of-fold predictions?", ["To save time", "So base models predict rows they have not seen (no leakage)",
                   "To add more features", "To shrink the dataset"], 1, "Otherwise the meta-model would over-trust base models that memorised the training rows.")
    finish(MID, flags().get("m5", False))
