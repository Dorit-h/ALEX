import time
import datetime
import streamlit as st

import rag


def setup_lecture(duration_minutes, interval_seconds):
    st.session_state.start_time = datetime.datetime.now()
    st.session_state.end_time = st.session_state.start_time + datetime.timedelta(minutes=duration_minutes)
    st.session_state.interval_seconds = interval_seconds
    st.session_state.current_time = st.session_state.start_time
    st.session_state.time_elapsed = datetime.timedelta(minutes=52)
    st.session_state.time_remaining = st.session_state.end_time - st.session_state.current_time

def update():
    st.session_state.current_time = datetime.datetime.now()
    st.session_state.time_elapsed = st.session_state.time_elapsed + datetime.timedelta(seconds=1)
    st.session_state.time_remaining = st.session_state.end_time - st.session_state.current_time


@st.fragment(run_every=1)
def time_display():
    update()
    st.write(f"Time Elapsed: {st.session_state.time_elapsed}")
    st.write("Currently on slide:", rag.Rag().get_current_slide(st.session_state.time_elapsed, "I2DL", "l02"))