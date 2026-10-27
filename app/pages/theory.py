"""Theory Library."""
import streamlit as st

from app.game import state
from app.game.content import THEORY
from src import content as C

state.init()
st.subheader("📚 Theory Library")
menu, body = st.columns([1, 3])
with menu:
    topic = st.radio("Topics", list(THEORY), key="theory_topic", label_visibility="collapsed")
with body:
    md, dot = THEORY[topic]
    st.markdown(f"## {topic}")
    if topic == "Interview Q&A":
        for cat, q, a in C.INTERVIEW_QA:
            with st.expander(f"{q}  ·  _{cat}_"):
                st.markdown(a)
    else:
        st.markdown(md)
        if dot:
            st.graphviz_chart(dot)
    state.mark_topic_read(topic)
    st.caption(f"Topics read: {len(state.g()['read_topics'])}/{len(THEORY)} (read 6 for the Bookworm badge)")
