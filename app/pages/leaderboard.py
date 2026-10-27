"""Leaderboard (local demo players + you)."""
import pandas as pd
import streamlit as st

from app.game import state
from app.game.content import DEMO_PLAYERS

state.init()
st.subheader("🏆 Leaderboard")
me = state.g()["player"]
rows = [dict(Player=n, Score=s, Level=l) for n, s, l in DEMO_PLAYERS] + [dict(Player=f"{me} (you)", Score=state.g()["xp"], Level=state.level())]
df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
df.insert(0, "#", df.index + 1)
st.dataframe(df.style.apply(lambda r: ["background-color: rgba(124,92,255,0.35)" if "(you)" in str(r["Player"]) else "" for _ in r], axis=1),
             hide_index=True, width="stretch")
st.caption("The other players are demo entries - this app keeps no server-side scores. Your score lives in this browser session; "
           "use **Settings → Export progress** to save it.")
