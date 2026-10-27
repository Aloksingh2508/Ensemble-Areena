"""Quiz & Challenge: 5 random questions, +50 XP per first-time correct answer."""
import random

import streamlit as st

from app.game import state
from app.game.content import QUIZ, QUIZ_LENGTH

state.init()
st.subheader("📝 Quiz & Challenge")
BANK = {q[0]: q for q in QUIZ}

TOPICS = {
    1: "🗳️ Hard Voting",
    2: "🗳️ Soft Voting",
    3: "🌲 Bagging & Variance",
    4: "🚀 Boosting Fundamentals",
    5: "🌲 Bootstrap Sampling",
    6: "🌲 Out-of-Bag (OOB)",
    7: "🌲 Random Forest",
    8: "📊 Overfitting & Generalization",
    9: "🚀 Decision Stumps & Bias",
    10: "🚀 AdaBoost Sample Weights",
    11: "🚀 Gradient Boosting Residuals",
    12: "🚀 Learning Rate Trade-offs",
    13: "🧠 Stacking & Leakage Prevention",
    14: "🧠 Meta-Learner Role",
    15: "📊 Ensemble Guarantees",
    16: "📊 Pipelines & Data Leakage",
    17: "📈 Evaluation Metrics (Recall)",
    18: "🤝 Model Diversity",
}


def _start() -> None:
    st.session_state["quiz"] = dict(
        ids=random.sample(list(BANK), QUIZ_LENGTH),
        i=0,
        score=0,
        submitted=False,
        last_ok=None,
        done=False,
        swaps=0,
    )
    st.session_state["quiz_choice"] = None
    st.session_state["quiz_notice"] = None


def _restart() -> None:
    _start()
    st.session_state["quiz_notice"] = "🔄 Quiz restarted with a fresh set of questions!"


def _swap() -> None:
    qz = st.session_state.get("quiz")
    if not qz or qz.get("done") or qz.get("submitted"):
        return
    current_qid = qz["ids"][qz["i"]]
    # Available questions not currently in this quiz set
    available = [qid for qid in BANK if qid not in qz["ids"]]
    if not available:
        available = [qid for qid in BANK if qid != current_qid]

    if available:
        new_qid = random.choice(available)
        qz["ids"][qz["i"]] = new_qid
        qz["swaps"] = qz.get("swaps", 0) + 1
        qz["submitted"] = False
        qz["last_ok"] = None
        st.session_state["quiz_choice"] = None
        new_topic = TOPICS.get(new_qid, "General ML")
        st.session_state["quiz_notice"] = f"🔀 Question #{qz['i'] + 1} swapped! New topic: **{new_topic}**."


def _submit() -> None:
    qz = st.session_state["quiz"]
    qid, _, options, answer, _ = BANK[qz["ids"][qz["i"]]]
    choice = st.session_state.get("quiz_choice")
    if choice is None:
        return
    ok = options.index(choice) == answer
    qz.update(submitted=True, last_ok=ok)
    state.record_answer(ok)
    if ok:
        qz["score"] += 1
        if qid not in state.g()["solved"]:
            state.g()["solved"].append(qid)
            state.add_xp(50, "- correct answer")


def _next() -> None:
    qz = st.session_state["quiz"]
    qz["i"] += 1
    qz["submitted"], qz["last_ok"] = False, None
    st.session_state["quiz_choice"] = None
    if qz["i"] >= QUIZ_LENGTH:
        qz["done"] = True
        state.g()["quiz_best"] = max(state.g()["quiz_best"], qz["score"])
        state.unlock_check()


if "quiz" not in st.session_state:
    solved_count = len(state.g()["solved"])
    st.markdown(
        "Answer **5 random questions** drawn from our ML question bank. "
        "Each question you get right for the first time earns **+50 XP**, and consecutive correct answers build your 🔥 streak."
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Bank Size", f"{len(BANK)} Questions")
    col_b.metric("Solved by You", f"{solved_count}/{len(BANK)}")
    col_c.metric("Current Streak", f"🔥 {state.g()['streak']}")

    st.markdown("💡 *Feature: You can now use **🔀 Swap Question** anytime during the quiz to exchange a tricky question for a fresh one from the bank, or **🔄 Restart Quiz** to get a new set.*")
    st.button("Start quiz", type="primary", key="quiz_start", on_click=_start)
    st.stop()

qz = st.session_state["quiz"]
if qz["done"]:
    st.success(f"Quiz finished! Score: **{qz['score']}/{QUIZ_LENGTH}**")
    if qz["score"] == QUIZ_LENGTH:
        st.balloons()
    swaps_used = qz.get("swaps", 0)
    st.markdown(f"Current streak: 🔥 **{state.g()['streak']}** · Total XP: **{state.g()['xp']}** · Swaps used: **{swaps_used}**")
    st.button("Play again", type="primary", key="quiz_again", on_click=_start)
    st.stop()

qid, text, options, answer, why = BANK[qz["ids"][qz["i"]]]
topic = TOPICS.get(qid, "General ML")
unused_in_bank = len(BANK) - len(qz["ids"])

# Header bar with progress, topic tag, and streak/swap counters
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.progress((qz["i"] + 1) / QUIZ_LENGTH, text=f"Question {qz['i'] + 1}/{QUIZ_LENGTH} · {topic}")
with h_col2:
    swaps_used = qz.get("swaps", 0)
    st.caption(f"🔥 Streak: **{state.g()['streak']}** | 🔀 Swaps: **{swaps_used}**")

notice = st.session_state.pop("quiz_notice", None)
if notice:
    st.info(notice, icon="ℹ️")

st.markdown(f"### {text}")
st.radio("Choose an answer", options, index=None, key="quiz_choice", disabled=qz["submitted"], label_visibility="collapsed")

if not qz["submitted"]:
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        st.button("Submit Answer", type="primary", key="quiz_submit", on_click=_submit, width="stretch")
    with btn_col2:
        st.button("🔀 Swap Question", key="quiz_swap", on_click=_swap, help=f"Exchange this question with one of the {unused_in_bank} unused questions in the bank", width="stretch")
    with btn_col3:
        st.button("🔄 Restart Quiz", key="quiz_restart", on_click=_restart, help="Restart the quiz with a fresh set of questions", width="stretch")
else:
    if qz["last_ok"]:
        st.success(f"✅ Correct! {why}")
    else:
        st.error(f"❌ Not quite. Correct answer: **{options[answer]}**. {why}")
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        st.button("Next Question" if qz["i"] < QUIZ_LENGTH - 1 else "Finish", type="primary", key="quiz_next", on_click=_next, width="stretch")
    with btn_col2:
        st.button("🔄 Restart Quiz", key="quiz_restart_submitted", on_click=_restart, help="Restart the quiz with a fresh set of questions", width="stretch")

