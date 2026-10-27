"""Helpers shared by all missions."""
import streamlit as st

from src.data import SplitData, make_demo_data, split_data
from app.game import state


def arena_data(noise: float = 0.3, seed: int = 7):
    """Small 2-D 'two moons' data used by Levels 2-5 (so decision boundaries can be drawn)."""
    ds = make_demo_data(400, noise, seed)
    return ds, split_data(ds, 0.3, seed)


def flags() -> dict:
    return state.g()["flags"]


def check_question(mid: int, question: str, options: list[str], answer: int, explain: str) -> bool:
    """Multiple-choice checkpoint. Returns True once passed (stored in progress)."""
    s = state.g()
    st.markdown(f"**🎯 Checkpoint:** {question}")
    if mid in s["checks"]:
        st.success("✅ Checkpoint passed. " + explain)
        return True
    choice = st.radio("Your answer", options, index=None, key=f"chk_{mid}", label_visibility="collapsed")
    if st.button("Submit answer", key=f"chk_btn_{mid}"):
        if choice is None:
            st.warning("Pick an answer first.")
        else:
            ok = options.index(choice) == answer
            state.record_answer(ok)
            if ok:
                s["checks"].append(mid)
                st.success("✅ Correct! " + explain)
                return True
            st.error("Not quite - think about it and try again.")
    return False


def objective(done: bool, text: str) -> None:
    st.markdown(("✅ " if done else "⬜ ") + f"**Objective:** {text}")


def finish(mid: int, objective_done: bool) -> None:
    """Complete the mission when objective + checkpoint are both done."""
    s = state.g()
    if mid in s["completed"]:
        st.success("🏅 Mission completed!")
        return
    if objective_done and mid in s["checks"]:
        if state.complete_mission(mid):
            st.balloons()
            st.success("🎉 Mission complete! Reward added to your score.")
    else:
        st.info("Finish the objective and pass the checkpoint to complete this mission.")
