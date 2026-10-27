"""Missions: pick a level card, then play it."""
import streamlit as st

from app import style
from app.game import state
from app.game.content import MISSIONS
from app.missions import level1, level2, level3, level4, level5, level6

state.init()
VIEWS = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5, 6: level6}

st.subheader("🎯 Missions")
if "active_mission" not in st.session_state:
    st.session_state["active_mission"] = state.level()

for col, m in zip(st.columns(len(MISSIONS)), MISSIONS):
    with col:
        st.markdown(style.mission_card(m), unsafe_allow_html=True)
        locked = state.status(m["id"]) == "locked"
        if st.button("Open" if not locked else "Locked", key=f"open_{m['id']}", disabled=locked, width="stretch"):
            st.session_state["active_mission"] = m["id"]

mid = st.session_state["active_mission"]
if state.status(mid) == "locked":
    mid = st.session_state["active_mission"] = state.level()
m = MISSIONS[mid - 1]
st.divider()
st.markdown(f"## {m['icon']} Level {m['id']} · {m['title']}")
st.caption(f"{m['sub']} · Reward: +{m['reward']} XP")
VIEWS[mid].render()
