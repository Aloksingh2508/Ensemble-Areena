"""ML Battle Arena - entry point.   Run with:  streamlit run app/main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` and `app` importable

import streamlit as st  # noqa: E402

from app import style  # noqa: E402
from app.game import state  # noqa: E402

st.set_page_config(page_title="Ensemble Learning - ML Battle Arena", page_icon="🎮", layout="wide")
state.init()
style.inject()
style.header()

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/missions.py", title="Missions", icon="🎯"),
    st.Page("pages/playground.py", title="Algorithm Playground", icon="🧪"),
    st.Page("pages/theory.py", title="Theory Library", icon="📚"),
    st.Page("pages/quiz.py", title="Quizzes", icon="📝"),
    st.Page("pages/leaderboard.py", title="Leaderboard", icon="🏆"),
    st.Page("pages/achievements.py", title="Achievements", icon="⭐"),
    st.Page("pages/settings.py", title="Settings", icon="⚙️"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.divider()
    style.player_card(compact=True)
nav.run()
