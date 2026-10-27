"""Game Dashboard (Home)."""
import streamlit as st

from app import style
from app.game import state
from app.game.content import MISSIONS

state.init()
st.subheader("🎮 Game Dashboard")
left, right = st.columns([2, 3])
with left:
    style.player_card()
with right:
    s = state.g()
    done = len(s["completed"])
    if done == len(MISSIONS):
        msg = f"You are an ML Ninja, {state.safe_name()}! Replay any mission or climb the leaderboard."
    elif s["xp"] == 0:
        msg = f"Welcome, {state.safe_name()}! Start with Level 1 to earn your first XP."
    else:
        msg = f"Great job, {state.safe_name()}! Keep learning and unlock the next level!"
    st.markdown(f'<div class="banner">🚩 {msg}</div>', unsafe_allow_html=True)
    style.link("pages/missions.py", "Continue your mission →", "🎯")

st.markdown("### Your Missions")
for col, m in zip(st.columns(len(MISSIONS)), MISSIONS):
    col.markdown(style.mission_card(m), unsafe_allow_html=True)

st.divider()
a, b, c = st.columns(3)
with a:
    style.link("pages/playground.py", "Open the Algorithm Playground", "🧪")
with b:
    style.link("pages/quiz.py", "Take a quick quiz (+50 XP per answer)", "📝")
with c:
    style.link("pages/theory.py", "Read the Theory Library", "📚")
st.info("**Honest note:** ensembles often help but are *not guaranteed* to beat a single model - the missions let you test that yourself.")
