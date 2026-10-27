"""Level 4 - Voting Arena: predict the hard / soft vote yourself."""
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.game import state
from app.missions.common import arena_data, flags

MID, NEEDED = 4, 3


def _models():
    """Three base models + their test-set probabilities (cached per session)."""
    if "m4_cache" not in st.session_state:
        ds, split = arena_data(noise=0.35)
        ms = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
              "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=1),
              "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=3, random_state=1)}
        P = np.stack([m.fit(split.X_train, split.y_train).predict_proba(split.X_test) for m in ms.values()])  # (3, n, 2)
        hard = (np.argmax(P, axis=2).sum(axis=0) >= 2).astype(int)
        soft = np.argmax(P.mean(axis=0), axis=1)
        st.session_state["m4_cache"] = dict(names=list(ms), P=P, disagree=np.where(hard != soft)[0], n=P.shape[1])
    return st.session_state["m4_cache"]


def _new_round() -> None:
    c = _models()
    s = st.session_state.setdefault("m4", dict(counter=0))
    s["counter"] += 1
    rng = np.random.RandomState(s["counter"])
    pool = c["disagree"] if (len(c["disagree"]) and rng.rand() < 0.5) else np.arange(c["n"])
    s.update(idx=int(rng.choice(pool)), kind=str(rng.choice(["Hard", "Soft"])), submitted=False, correct=None)
    st.session_state["m4_choice"] = None


def _submit() -> None:
    s, c = st.session_state["m4"], _models()
    choice = st.session_state.get("m4_choice")
    if choice is None:
        return
    P = c["P"][:, s["idx"], :]
    answer = int(np.argmax(P.mean(axis=0))) if s["kind"] == "Soft" else int(np.argmax(P, axis=1).sum() >= 2)
    s["correct"] = (int(choice[-1]) == answer)
    s["answer"], s["submitted"] = answer, True
    state.record_answer(s["correct"])
    if s["correct"]:
        state.add_xp(25, "- correct vote")
        state.g()["voting_correct"] += 1
        if state.g()["voting_correct"] >= NEEDED:
            flags()["m4"] = True
            if state.complete_mission(MID):
                st.balloons()


def render() -> None:
    st.markdown("Three base models each make a prediction. **Hard voting** takes the majority label; **soft voting** averages the probabilities. "
                f"Predict the final answer yourself. **Objective:** get **{NEEDED} rounds right**.")
    c = _models()
    if "m4" not in st.session_state:
        _new_round()
    s = st.session_state["m4"]
    P = c["P"][:, s["idx"], :]
    left, mid, right = st.columns(3)
    with left:
        st.markdown("##### Base models")
        for name, p in zip(c["names"], P):
            st.markdown(f'<div class="model-card"><b>{name}</b><br>Prediction: <b>Class {int(np.argmax(p))}</b><br>'
                        f'<small>Prob: [{p[0]:.2f}, {p[1]:.2f}]</small></div>', unsafe_allow_html=True)
    with mid:
        st.markdown("##### Your turn!")
        st.write(f"What will be the final prediction using **{s['kind']} Voting**?")
        st.radio("Answer", ["A. Class 0", "B. Class 1"], index=None, key="m4_choice", disabled=s["submitted"], label_visibility="collapsed")
        if not s["submitted"]:
            st.button("Submit Answer", type="primary", key="m4_submit", on_click=_submit)
        else:
            st.button("Next round ➡", key="m4_next", on_click=_new_round)
    with right:
        st.markdown("##### Final results")
        if s["submitted"]:
            view = st.radio("Voting type", ["Hard", "Soft"], index=0 if s["kind"] == "Hard" else 1, horizontal=True, key="m4_view")
            hv = int(np.argmax(P, axis=1).sum() >= 2)
            if view == "Hard":
                votes = np.argmax(P, axis=1)
                st.success(f"Final prediction: **Class {hv}**")
                st.caption(f"Hard voting takes the majority class: {int((votes == hv).sum())} of 3 models predicted Class {hv}.")
            else:
                avg = P.mean(axis=0)
                st.success(f"Final prediction: **Class {int(np.argmax(avg))}**")
                st.caption(f"Soft voting averages probabilities: [{avg[0]:.2f}, {avg[1]:.2f}] → Class {int(np.argmax(avg))}.")
            (st.success if s["correct"] else st.error)("Correct! +25 XP" if s["correct"] else f"Not this time - the answer was Class {s['answer']}.")
        else:
            st.info("Submit your answer to reveal the results.")
    done = state.g()["voting_correct"]
    st.progress(min(done / NEEDED, 1.0), text=f"Correct rounds: {done}/{NEEDED}")
    if MID in state.g()["completed"]:
        st.success("🏅 Mission completed! Keep playing for extra XP and streak points.")
