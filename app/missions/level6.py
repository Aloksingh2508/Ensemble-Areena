"""Level 6 - Final Championship: pick the best algorithm for a dataset (6-step wizard)."""
import streamlit as st

from src import plots
from src.data import BREAST_CANCER, IRIS, SYNTHETIC, load_dataset, split_data
from src.models import ADABOOST, BAGGING, FOREST, GRADBOOST, STACKING, TREE, VOTING
from src.training import results_table, run_experiment
from app.game import state
from src.ui import show_df, show_plotly

MID = 6
STEPS = ["Dataset", "Select Models", "Configure", "Train", "Compare", "Choose"]
MODELS = [TREE, FOREST, BAGGING, ADABOOST, GRADBOOST, VOTING, STACKING]
TOLERANCE = 0.005


def _s() -> dict:
    return st.session_state.setdefault("champ", dict(step=0, dataset=BREAST_CANCER, models=[FOREST, TREE], n_est=50, depth=5, run=None, result=None))


def _go(delta: int) -> None:
    c = _s()
    c["step"] = max(0, min(len(STEPS) - 1, c["step"] + delta))


def _restart() -> None:
    st.session_state.pop("champ", None)


def _params(name: str, c: dict) -> dict:
    p = {TREE: {"max_depth": c["depth"]}, FOREST: {"n_estimators": c["n_est"], "max_depth": c["depth"]},
         BAGGING: {"n_estimators": min(c["n_est"], 200), "max_depth": c["depth"]},
         ADABOOST: {"n_estimators": c["n_est"], "learning_rate": 0.5, "max_depth": 2},
         GRADBOOST: {"n_estimators": c["n_est"], "max_depth": min(c["depth"], 6)},
         VOTING: {"voting": "soft", "max_depth": c["depth"]}, STACKING: {"cv": 3, "rf_estimators": 30, "max_depth": c["depth"]}}
    return p[name]


def render() -> None:
    c = _s()
    st.markdown("Choose the **best algorithm** for the dataset. You will train several candidates and then lock in your pick. "
                "The winner is decided by **test F1-score**.")
    pills = "".join(f'<span class="pill {"now" if i == c["step"] else "done" if i < c["step"] else ""}">{i + 1} {n}</span>' for i, n in enumerate(STEPS))
    st.markdown(pills, unsafe_allow_html=True)
    step = c["step"]
    can_next = True
    if step == 0:
        c["dataset"] = st.selectbox("Select dataset", [BREAST_CANCER, IRIS, SYNTHETIC], index=[BREAST_CANCER, IRIS, SYNTHETIC].index(c["dataset"]), key="ch_ds")
    elif step == 1:
        c["models"] = st.multiselect("Choose 2 or more algorithms", MODELS, default=[m for m in c["models"] if m in MODELS], key="ch_models")
        can_next = len(c["models"]) >= 2
        if not can_next:
            st.warning("Select at least two algorithms.")
    elif step == 2:
        a, b = st.columns(2)
        c["n_est"] = a.slider("Number of estimators (ensembles)", 10, 200, c["n_est"], 10, key="ch_n")
        c["depth"] = b.slider("Max tree depth", 1, 12, c["depth"], key="ch_d")
    elif step == 3:
        st.write(f"Ready to train **{', '.join(c['models'])}** on **{c['dataset']}**.")
        if st.button("🏋️ Train all", type="primary", key="ch_train"):
            ds = load_dataset(c["dataset"], n_samples=600, n_features=6, seed=11)
            split = split_data(ds, 0.25, 11)
            with st.spinner("Training..."):
                results, errors = run_experiment({m: _params(m, c) for m in c["models"]}, split, 11)
            c["run"] = dict(results=results, errors=errors, split=split)
            state.count_training()
        can_next = bool(c["run"] and c["run"]["results"])
        if c["run"]:
            for n, e in c["run"]["errors"].items():
                st.error(f"{n}: {e}")
            st.success(f"Trained {len(c['run']['results'])} model(s).") if c["run"]["results"] else None
    elif step in (4, 5):
        run = c["run"]
        if not run or not run["results"]:
            st.warning("Train some models first (step 4).")
            can_next = False
        else:
            df = results_table(run["results"])
            show_df(df.style.format({m: "{:.3f}" for m in plots.METRICS} | {"Train time (s)": "{:.4f}"}))
            show_plotly(plots.metric_bars(df), key=f"ch_bars_{step}")
            if step == 5:
                pick = st.selectbox("🏆 Your champion", list(df["Algorithm"]), index=None, placeholder="Choose an algorithm", key="ch_pick")
                if st.button("Lock in my choice", type="primary", key="ch_lock", disabled=pick is None):
                    f1 = dict(zip(df["Algorithm"], df["F1-score"]))
                    best_name, best = max(f1.items(), key=lambda kv: kv[1])
                    c["result"] = dict(pick=pick, best=best_name, ok=f1[pick] >= best - TOLERANCE)
                    state.record_answer(c["result"]["ok"])
                    if c["result"]["ok"] and state.complete_mission(MID):
                        st.balloons()
                res = c["result"]
                if res:
                    if res["ok"]:
                        st.success(f"🏆 Champion! **{res['pick']}** is (within {TOLERANCE}) the best F1 on this test set.")
                    else:
                        st.error(f"Not this time - **{res['best']}** had the best F1. Try different settings or a new dataset!")
                    st.caption("Remember: on a different split another model might win. That is why we compare on the same split and, ideally, several splits.")
                    st.button("🔁 Play again", key="ch_again", on_click=_restart)
    if state.status(MID) == "completed" and not c["result"]:
        st.success("🏅 You already completed this mission - feel free to play again!")
    nav = st.columns([1, 1, 4])
    nav[0].button("◀ Back", key="ch_back", disabled=step == 0, on_click=_go, args=(-1,))
    if step < len(STEPS) - 1:
        nav[1].button("Next ▶", key="ch_next", type="primary", disabled=not can_next, on_click=_go, args=(1,))
