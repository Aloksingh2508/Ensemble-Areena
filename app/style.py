"""Dark 'arena' styling and small HTML building blocks."""
import streamlit as st

from .game import state
from .game.content import ACHIEVEMENTS, MISSIONS, XP_GOAL

CSS = """
<style>
.arena-title{font-size:2.1rem;font-weight:900;line-height:1.05;letter-spacing:.5px;
  background:linear-gradient(90deg,#7C9CFF,#B28CFF);-webkit-background-clip:text;color:transparent}
.arena-title span{background:linear-gradient(90deg,#FFB347,#FF7A18);-webkit-background-clip:text;color:transparent}
.arena-tag{color:#8FA3FF;letter-spacing:2px;font-size:.9rem}
.perk{text-align:center;font-size:.8rem;color:#C9D3FF}.perk b{display:block;font-size:1.5rem}
.panel{background:#131B3D;border:1px solid #26336B;border-radius:14px;padding:14px 16px;margin-bottom:10px}
.stat{background:#10173A;border:1px solid #26336B;border-radius:10px;padding:8px 12px;text-align:center}
.stat small{color:#8FA3FF;display:block}.stat b{font-size:1.5rem}
.banner{background:linear-gradient(90deg,#3A2A9C,#1F5FBF);border-radius:12px;padding:14px 18px;font-weight:600;margin:8px 0 14px}
.mcard{border-radius:12px;padding:12px;border:1px solid #26336B;background:#10173A;min-height:128px}
.mcard.completed{border-color:#2FB36B;background:#0F2A23}
.mcard.in_progress{border-color:#7C5CFF;background:#1A1850;box-shadow:0 0 14px rgba(124,92,255,.45)}
.mcard.locked{opacity:.55}
.mcard .lv{font-size:.75rem;color:#8FA3FF}.mcard .tt{font-weight:800;margin:2px 0}.mcard .sb{font-size:.78rem;color:#B9C4F5}
.mcard .stt{margin-top:8px;font-size:.78rem;font-weight:700}
.badge{border-radius:12px;padding:12px;border:1px solid #26336B;background:#10173A;text-align:center;min-height:132px}
.badge.on{border-color:#F5B301;background:#2A2410}.badge.off{opacity:.45;filter:grayscale(1)}
.badge .ic{font-size:2rem}.badge small{color:#B9C4F5}
.boot-row{margin:4px 0}.dot{display:inline-block;width:20px;height:20px;border-radius:50%;margin:1px 2px;text-align:center;
  font-size:.7rem;line-height:20px;border:2px solid #7C9CFF}.dot.on{background:#7C9CFF;color:#0B1026}.dot.off{opacity:.35}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;border:1px solid #26336B;margin-right:6px;font-size:.8rem}
.pill.now{background:#7C5CFF;border-color:#7C5CFF}.pill.done{background:#1F6F4A;border-color:#2FB36B}
.model-card{background:#10173A;border:1px solid #26336B;border-radius:12px;padding:10px 12px;margin-bottom:8px}
.me{background:#1A1850}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    left, right = st.columns([2, 3])
    with left:
        st.markdown('<div class="arena-title">ENSEMBLE LEARNING<br><span>ML BATTLE ARENA</span></div>'
                    '<div class="arena-tag">Learn • Play • Build • Become an ML Ninja</div>', unsafe_allow_html=True)
    with right:
        perks = [("🎮", f"{len(MISSIONS)} Levels"), ("🧠", "Interactive Visualizations"), ("🏆", "XP & Achievements"),
                 ("📖", "Theory Library"), ("📝", "Quizzes & Challenges"), ("📊", "Leaderboard")]
        for col, (ic, label) in zip(st.columns(len(perks)), perks):
            col.markdown(f'<div class="perk"><b>{ic}</b>{label}</div>', unsafe_allow_html=True)


def player_card(compact: bool = False) -> None:
    s = state.g()
    xp = min(s["xp"], XP_GOAL)
    st.markdown(f'<div class="panel"><b>🧑‍💻 Player: {state.safe_name()}</b><br>'
                f'<small>Level {state.level()} - {state.title()}</small></div>', unsafe_allow_html=True)
    st.progress(state.xp_fraction(), text=f"{xp} / {XP_GOAL} XP" + ("+" if s["xp"] > XP_GOAL else ""))
    if not compact:
        a, b, c = st.columns(3)
        a.markdown(f'<div class="stat"><small>Total Score</small><b>{s["xp"]}</b></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="stat"><small>Achievements</small><b>🏆 {len(s["achievements"])}/{len(ACHIEVEMENTS)}</b></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="stat"><small>Current Streak</small><b>🔥 {s["streak"]}</b></div>', unsafe_allow_html=True)


LABEL = {"completed": "✔ Completed", "in_progress": "▶ In Progress", "locked": "🔒 Locked"}
COLOR = {"completed": "#2FB36B", "in_progress": "#9C8CFF", "locked": "#8A93B8"}


def mission_card(m: dict) -> str:
    stt = state.status(m["id"])
    return (f'<div class="mcard {stt}"><div class="lv">Level {m["id"]}</div><div class="tt">{m["icon"]} {m["title"]}</div>'
            f'<div class="sb">{m["sub"]}</div><div class="stt" style="color:{COLOR[stt]}">{LABEL[stt]} · +{m["reward"]} XP</div></div>')


def link(path: str, label: str, icon: str = "") -> None:
    """page_link that degrades gracefully when a page runs outside st.navigation."""
    from streamlit.errors import StreamlitPageNotFoundError

    try:
        st.page_link(path, label=label, icon=icon or None)
    except StreamlitPageNotFoundError:
        st.caption(f"{icon} {label}")
