"""Player progress: XP, levels, streaks, achievements (kept in st.session_state)."""
from __future__ import annotations

import copy
import html
import json

import streamlit as st

from .content import ACHIEVEMENTS, MISSIONS, QUIZ_LENGTH, TITLES, XP_GOAL

DEFAULTS = {
    "player": "Player", "xp": 0, "completed": [], "streak": 0, "best_streak": 0, "achievements": [],
    "read_topics": [], "solved": [], "quiz_best": 0, "trained": 0, "checks": [], "flags": {}, "voting_correct": 0,
}
_LISTS = ("completed", "achievements", "read_topics", "solved", "checks")

CONDITIONS = {
    "first_step": lambda s: 1 in s["completed"],
    "forest_explorer": lambda s: 2 in s["completed"],
    "boost_hero": lambda s: 3 in s["completed"],
    "voting_veteran": lambda s: 4 in s["completed"],
    "stack_master": lambda s: 5 in s["completed"],
    "champion": lambda s: 6 in s["completed"],
    "quiz_master": lambda s: s["quiz_best"] >= 4,
    "perfect_score": lambda s: s["quiz_best"] >= QUIZ_LENGTH,
    "streak_starter": lambda s: s["best_streak"] >= 3,
    "hot_streak": lambda s: s["best_streak"] >= 5,
    "theory_reader": lambda s: len(s["read_topics"]) >= 6,
    "experimenter": lambda s: s["trained"] >= 5,
}


def init() -> None:
    if "game" not in st.session_state:
        st.session_state["game"] = copy.deepcopy(DEFAULTS)


def g() -> dict:
    init()
    return st.session_state["game"]


def safe_name() -> str:
    """HTML-escaped player name (safe to place inside unsafe_allow_html blocks)."""
    return html.escape(g()["player"])


# --------------------------------------------------------------- derived
def level() -> int:
    return min(len(g()["completed"]) + 1, len(MISSIONS))


def title() -> str:
    return TITLES[level()]


def status(mission_id: int) -> str:
    s = g()
    if mission_id in s["completed"]:
        return "completed"
    if mission_id == 1 or (mission_id - 1) in s["completed"]:
        return "in_progress"
    return "locked"


def xp_fraction() -> float:
    return min(g()["xp"] / XP_GOAL, 1.0)


# ---------------------------------------------------------------- actions
def add_xp(amount: int, why: str = "") -> None:
    g()["xp"] += int(amount)
    st.toast(f"+{amount} XP {why}".strip(), icon="⭐")


def unlock_check() -> None:
    s = g()
    for a in ACHIEVEMENTS:
        if a["id"] not in s["achievements"] and CONDITIONS[a["id"]](s):
            s["achievements"].append(a["id"])
            add_xp(a["xp"], f"- Achievement: {a['title']}")
            st.toast(f"Achievement unlocked: {a['title']}", icon="🏅")


def complete_mission(mission_id: int) -> bool:
    """Mark a mission as done (once) and pay its reward. Returns True if newly completed."""
    s = g()
    if mission_id in s["completed"]:
        return False
    s["completed"].append(mission_id)
    reward = next(m["reward"] for m in MISSIONS if m["id"] == mission_id)
    add_xp(reward, f"- Level {mission_id} complete")
    unlock_check()
    return True


def record_answer(correct: bool) -> None:
    s = g()
    if correct:
        s["streak"] += 1
        s["best_streak"] = max(s["best_streak"], s["streak"])
    else:
        s["streak"] = 0
    unlock_check()


def count_training(n: int = 1) -> None:
    g()["trained"] += n
    unlock_check()


def mark_topic_read(topic: str) -> None:
    s = g()
    if topic not in s["read_topics"]:
        s["read_topics"].append(topic)
        unlock_check()


def set_name(name: str) -> None:
    g()["player"] = (name or "").strip()[:20] or "Player"


def reset() -> None:
    st.session_state["game"] = copy.deepcopy(DEFAULTS)
    for k in [k for k in st.session_state if k != "game"]:
        del st.session_state[k]


def export_progress() -> str:
    return json.dumps(g(), indent=2)


def import_progress(text: str) -> None:
    """Load a progress file, validating types so a bad file cannot corrupt the state."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Progress file must contain a JSON object.")
    new = copy.deepcopy(DEFAULTS)
    for key, default in DEFAULTS.items():
        if key not in data:
            continue
        v = data[key]
        if isinstance(default, bool) or not isinstance(v, type(default)):
            raise ValueError(f"Invalid value for '{key}'.")
        new[key] = v
    new["player"] = new["player"].strip()[:20] or "Player"
    for k in ("xp", "streak", "best_streak", "quiz_best", "trained", "voting_correct"):
        new[k] = max(0, min(int(new[k]), 10**6))
    new["completed"] = [int(m) for m in new["completed"] if m in range(1, len(MISSIONS) + 1)]
    valid_ach = {a["id"] for a in ACHIEVEMENTS}
    new["achievements"] = [a for a in new["achievements"] if a in valid_ach]
    st.session_state["game"] = new
