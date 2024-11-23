import datetime
import streamlit as st
st.set_page_config(page_title="Lecture Selector", layout="wide")
import random
from openai import OpenAI
import time_keeper
from rag import Rag


# Streamed response emulator
def response_generator(user_input: str):

    rag = Rag()
    return rag.run(user_input, lecture="I2DL", lecture_id="l02")
    

# Add the logo to the main page
logo_url = "https://softwarecampus.de/wp-content/uploads/logo-partner-software-campus-tum.webp"

# Inject custom CSS for blue sidebars
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #FFFFFF 20%, white 20%, white 80%, #8ACEF1 80%);
        }
        .sidebar-logo-container {
            display: flex; /* Use flexbox for alignment */
            justify-content: center; /* Center horizontally */
            align-items: center; /* Center vertically */
        }
        .sidebar-logo-container img {
            width: 200px; /* Adjust the width */
            border-radius: 10px; /* Optional rounded corners */
            background-color: white; /* Optional background */
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow */
        }
        /* Styling for chat input */
        div[data-testid="stChatInput"] {
            background-color: #D9E8F1 !important; /* Set the background color */
            border-radius: 10px; /* Optional rounded corners */
            padding: 0px; /* Optional padding for spacing */
        }
    </style>
    """,
    unsafe_allow_html=True
)

## Sidebar

with st.sidebar:
    # Centered logo at the top
    st.markdown(
        f"""
        <div class="sidebar-logo-container">
            <img src="{logo_url}" alt="TUM Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("***")

for i in range(10):
    st.sidebar.markdown("\n")

# Add a session state to track whether live selection is active
if "liveselection" not in st.session_state:
    st.session_state.liveselection = False

# Top bar with dropdown and slider button
col1, col2 = st.columns([4, 1])
with st.sidebar:
    live_lecture = st.toggle("Live Lecture", value=True)
    st.session_state.liveselection = live_lecture

with st.sidebar:
    selected_course = st.selectbox(
        "Select a Course:",
        [
            "Introduction to Computer Science",
            "Data Structures and Algorithms",
            "Operating Systems",
            "Introduction to Deep Learning",
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

for i in range(10):
    st.sidebar.markdown("\n")

st.sidebar.markdown("***")
with st.sidebar:
    time_keeper.setup_lecture(120, 5)
    time_keeper.time_display()   

## Main

# Title and Subtitle
st.title("ALEX")
st.subheader("Augmented Lecture Explainer")

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
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxuutX8HduKl2eiBeqSWo1VdXcOS9UxzsKhQ&s"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    response = ""
    
    with st.chat_message("assistant", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF9mciO08VZ5zdZbfLqlLarccmeMZLByJ_9w&s"):
        with st.spinner("Getting your answer..."):
            response = response_generator(prompt)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

