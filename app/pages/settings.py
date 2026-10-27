"""Settings: player name, save / load progress, reset."""
import streamlit as st

from app.game import state

state.init()
st.subheader("⚙️ Settings")
name = st.text_input("Player name (max 20 characters)", value=state.g()["player"], max_chars=20, key="set_name")
if name != state.g()["player"]:
    state.set_name(name)

st.markdown("#### Save / load progress")
st.download_button("⬇ Export progress (JSON)", state.export_progress(), file_name="ml_battle_arena_progress.json", mime="application/json")
up = st.file_uploader("Import a progress file", type="json", key="set_upload")
if up is not None and st.button("Load progress", key="set_load"):
    try:
        state.import_progress(up.getvalue().decode("utf-8"))
        st.success("Progress loaded.")
        st.rerun()
    except (ValueError, UnicodeDecodeError) as exc:  # json.JSONDecodeError is a ValueError
        st.error(f"Could not load that file: {exc}")

st.markdown("#### Danger zone")
if st.checkbox("I understand this erases my XP, missions and achievements", key="set_confirm"):
    if st.button("Reset all progress", key="set_reset"):
        state.reset()
        st.rerun()
