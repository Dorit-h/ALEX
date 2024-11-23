import datetime
from io import StringIO
import sys
import streamlit as st
st.set_page_config(page_title="Lecture Selector", layout="wide")

if "time_elapsed" not in st.session_state:
    st.session_state.time_elapsed = datetime.timedelta(minutes=52)
import random
from openai import OpenAI
import time_keeper
from rag import Rag


# Streamed response emulator
def response_generator(user_input: str):

    rag = Rag()
    return rag.run(user_input, lecture="I2DL", lecture_id="l02")


# Add the logo to the main page
logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTJXx0UCQv_9GASiIk2qqtv4hT7m9Mp1F-fQ&s"

# Inject custom CSS for blue sidebars
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #FFFFFF 20%, white 20%, white 80%, #8ACEF1 80%);
        }
        .sidebar-logo-container {
            display: flex; /* Use flexbox for alignment */
            justify-content: right; /* Center horizontally */
            align-items: center; /* Center vertically */
        }
        .sidebar-logo-container img {
            width: 100px; /* Adjust the width */
            padding: 0px;
        }
        /* Styling for chat input */
        div[data-testid="stChatInput"] {
            background-color: #D9E8F1 !important; /* Set the background color */
            border-radius: 10px; /* Optional rounded corners */
        }
        /* Styling for select boxes */
        div[data-baseweb="select"] > div {
            background-color: #D9E8F1 !important;
        }
        /* Change sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #0071BD; /* Light blue background */
        }
    </style>
    """,
    unsafe_allow_html=True
)

## Sidebar

for i in range(2):
    st.sidebar.markdown("\n")

# Add a session state to track whether live selection is active
if "liveselection" not in st.session_state:
    st.session_state.liveselection = False

# Top bar with dropdown and slider button
with st.sidebar:
    live_lecture = st.toggle("Live Lecture", value=True)
    st.session_state.liveselection = live_lecture

with st.sidebar:
    selected_course = st.selectbox(
        "Select a Course:",
        [
            "Introduction to Deep Learning",
            "Introduction to Computer Science",
            "Data Structures and Algorithms",
            "Operating Systems",
            "Machine Learning",
            "Artificial Intelligence",
            "Computer Networks",
            "Software Engineering",
            "Database Systems",
            "Computer Vision",
            "Natural Language Processing",
            "Compiler Construction",
            "Theory of Computation",
        ],
    )
    if not live_lecture:
        selected_lecture = st.selectbox(
            "Select a lecture:", [f"Lecture {i}" for i in range(1, 13)]
        )

for i in range(2):
    st.sidebar.markdown("\n")

if live_lecture:

    st.sidebar.markdown("***")
    with st.sidebar:
        time_keeper.time_display()   

## Main
col1, col2, col3 = st.columns(3)
with col1:
# Title and Subtitle
    st.title("ALEX")
    st.subheader("Augmented Lecture Explainer")
with col3:
    # Centered logo at the top
    #st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTJXx0UCQv_9GASiIk2qqtv4hT7m9Mp1F-fQ&s", width=100)
    st.markdown(
        f"""
        <div class="sidebar-logo-container">
            <img src="{logo_url}" alt="TUM Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )
    

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Assign avatar based on role
    avatar = (
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxuutX8HduKl2eiBeqSWo1VdXcOS9UxzsKhQ&s"
        if message["role"] == "user"
        else "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF9mciO08VZ5zdZbfLqlLarccmeMZLByJ_9w&s"
    )
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
def execute_python_code(code, code_type):
    codeOut = StringIO()
    codeErr = StringIO()
    sys.stdout = codeOut
    sys.stderr = codeErr
    exec(code)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    st.caption("Std. Error")
    st.code(codeErr.getvalue())
    st.caption("Std. Out")
    st.code(codeOut.getvalue())
    codeOut.close()
    codeErr.close()


if prompt := st.chat_input("Nice to meet you. Ask me about your lecture."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxuutX8HduKl2eiBeqSWo1VdXcOS9UxzsKhQ&s"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    response = ""
    
    with st.chat_message("assistant", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF9mciO08VZ5zdZbfLqlLarccmeMZLByJ_9w&s"):
        text_output = st.empty()
        with text_output:
            with st.spinner("Getting your answer..."):
                response = st.write_stream(response_generator(prompt))
        
        if "```" in response:
            
            with text_output.container():
                st.markdown(response.split("```")[0])
                code_segment: str = response.split("```")[1]
                code = code_segment.split("\n", 1)[1]
                code_type = code_segment.split("\n")[0]
                with st.expander("Show code"):
                    st.code(code)
                with st.expander("Output", True):
                    st.markdown("#### Code Output")
                    if code_type == "python":
                        execute_python_code(code, code_type)
                    else:
                        with st.container(border=True):
                            st.components.v1.html(code, height=600)
            st.markdown(response.split("```")[2])
        else:
            text_output.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
