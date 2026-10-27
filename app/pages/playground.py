"""Algorithm Playground / Live Visualization."""
import json
import time

import streamlit as st

from app.game import state
from src import ui
from src.models import ADABOOST, FOREST, TREE
from src.training import run_experiment

state.init()
st.subheader("🧪 Algorithm Playground - Live Visualization")
st.caption("Pick data and algorithms, tune parameters, press **Train Models**, then explore every tab.")

cfg = ui.dataset_controls("pg")
selected = ui.algorithm_selector("pg", [TREE, FOREST, ADABOOST])
signature = json.dumps({"cfg": cfg, "selected": selected}, sort_keys=True, default=str)

if st.button("🚀 Train Models", type="primary", key="train_button"):
    if not selected:
        st.warning("Please select at least one algorithm.")
    else:
        try:
            ds, split = ui.build_data(cfg)
        except ValueError as exc:
            st.error(f"Invalid data settings: {exc}")
        else:
            with st.spinner("Training models..."):
                results, errors = run_experiment(selected, split, cfg["seed"])
            st.session_state["last_run"] = dict(dataset=ds, split=split, results=results, errors=errors, seed=cfg["seed"],
                                                test_size=cfg["test_size"], selected=selected, signature=signature, id=time.time(), cfg=cfg)
            state.count_training()

run = ui.get_last_run()
if run is None:
    st.info("No results yet - choose algorithms and press **Train Models**.")
    st.stop()
for name, msg in run["errors"].items():
    st.error(f"**{name}** could not be trained: {msg}")
if not run["results"]:
    st.stop()
if run["signature"] != signature:
    st.warning("Settings changed since the last run - these results are from the **previous run**. Press **Train Models** to update.")

tabs = st.tabs(["Dataset", "Decision Boundary", "Individual Models", "Ensemble", "Metrics"])
with tabs[0]:
    ui.render_dataset(run, "pg_ds")
with tabs[1]:
    ui.render_boundaries(run, "pg_bd")
with tabs[2]:
    ui.render_predictions(run, "pg_pr", part="table")
    ui.render_confusions(run, "pg_cm")
with tabs[3]:
    ui.render_predictions(run, "pg_en", part="members")
    st.subheader("Boosting iterations")
    ui.render_boosting(run, "pg_bo")
with tabs[4]:
    ui.render_scores(run, "pg_sc")
    st.subheader("Feature importance")
    ui.render_importance(run, "pg_fi")
