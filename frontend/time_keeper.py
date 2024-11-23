import time
import datetime
import streamlit as st

import rag



def update():
    st.session_state.time_elapsed = st.session_state.time_elapsed + datetime.timedelta(seconds=1)


@st.fragment(run_every=1)
def time_display():
    update()
    st.pills(f"Lecture Timestamp:", options=[st.session_state.time_elapsed])
    st.pills("The Lecture is currently on slide:", options=[rag.Rag().get_current_slide(st.session_state.time_elapsed, "I2DL", "l02")])