"""Achievements wall."""
import streamlit as st

from app.game import state
from app.game.content import ACHIEVEMENTS

state.init()
st.subheader("⭐ Achievements")
got = state.g()["achievements"]
st.progress(len(got) / len(ACHIEVEMENTS), text=f"{len(got)}/{len(ACHIEVEMENTS)} unlocked")
cols = st.columns(4)
for i, a in enumerate(ACHIEVEMENTS):
    on = a["id"] in got
    cols[i % 4].markdown(f'<div class="badge {"on" if on else "off"}"><div class="ic">{a["icon"]}</div><b>{a["title"]}</b><br>'
                         f'<small>{a["desc"]}</small><br><b>{"+%d XP" % a["xp"] if on else "🔒 %d XP" % a["xp"]}</b></div>', unsafe_allow_html=True)
