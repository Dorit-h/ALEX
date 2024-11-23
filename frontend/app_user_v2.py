import streamlit as st
import random
from openai import OpenAI
import time_keeper
from rag import Rag

# Streamed response emulator
def response_generator(user_input: str):

    rag = Rag()
    return rag.run(user_input)
    
time_keeper.setup_lecture(120, 5)
time_keeper.time_display()

# Inject custom CSS for blue sidebars
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #FFFFFF 20%, white 20%, white 80%, #8ACEF1 80%);
        }
        .block-container {
            padding-top: 2rem;
            padding-left: 5%;
            padding-right: 5%;
            background-color: #89A8C2;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Add a session state to track whether live selection is active
if "liveselection" not in st.session_state:
    st.session_state.liveselection = False

# Top bar with dropdown and slider button
col1, col2 = st.columns([4, 1])
with col2:
    live_lecture = st.toggle("Live Lecture", value=True)
    st.session_state.liveselection = live_lecture

with col1:
    selected_course = st.selectbox(
        "Select a Course:",
        [
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

# Accept user input
if prompt := st.chat_input("Ask me anything."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxuutX8HduKl2eiBeqSWo1VdXcOS9UxzsKhQ&s"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    response = ""
    with st.chat_message("assistant", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF9mciO08VZ5zdZbfLqlLarccmeMZLByJ_9w&s"):
        st.write(response_generator(prompt))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
