"""Level 1 - Prediction Rookie: train a Decision Tree, learn accuracy basics."""
import streamlit as st

from src import plots
from src.data import BREAST_CANCER, load_dataset, split_data
from src.models import TREE
from src.training import train_and_evaluate
from app.game import state
from app.missions.common import check_question, finish, flags, objective
from src.ui import show_plotly

MID, TARGET = 1, 0.94


def render() -> None:
    st.markdown(
        "A **prediction** is a model's best guess for a new sample. **Accuracy** = correct predictions ÷ all predictions, "
        "but accuracy alone can hide mistakes, so we also look at **precision** (when I say *positive*, am I right?) and "
        "**recall** (of all real positives, how many did I find?).\n\n"
        "**Your task:** train a Decision Tree on the *Breast Cancer* dataset and reach at least **94% accuracy**."
    )
    ds = load_dataset(BREAST_CANCER)
    split = split_data(ds, 0.25, 42)
    a, b = st.columns(2)
    depth = a.slider("Tree depth", 1, 10, 1, key="m1_depth", help="Depth 1 = one yes/no question. Deeper = more questions.")
    leaf = b.slider("Min samples per leaf", 1, 20, 1, key="m1_leaf")
    if st.button("🏋️ Train Decision Tree", type="primary", key="m1_train"):
        st.session_state["m1_res"] = train_and_evaluate(TREE, {"max_depth": depth, "min_samples_leaf": leaf}, split, 42)
        state.count_training()
    r = st.session_state.get("m1_res")
    if r:
        cols = st.columns(4)
        for col, (k, v) in zip(cols, r.metrics.items()):
            col.metric(k, f"{v:.3f}")
        show_plotly(plots.confusion_heatmap(r.confusion, split.class_names, "Confusion matrix (test set)"), key="m1_cm")
        if r.metrics["Accuracy"] >= TARGET:
            flags()["m1"] = True
        else:
            st.info("Below 94% - try a different depth. Note that deeper is not always better!")
    objective(flags().get("m1", False), f"reach ≥ {TARGET:.0%} test accuracy")
    check_question(MID, "Which metric tells you how many of the REAL positive cases the model found?",
                   ["Precision", "Recall", "Training time", "Depth"], 1,
                   "Recall = true positives ÷ (true positives + false negatives).")
    finish(MID, flags().get("m1", False))
