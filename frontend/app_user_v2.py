import streamlit as st
import random
import time
from openai import OpenAI
import time_keeper
from rag import Rag

# Streamed response emulator
def response_generator(user_input: str):
    vlm_model = OpenAI(base_url="https://095kiew15yzv2e-8000.proxy.runpod.net/v1/", api_key="volker123")

    # return vlm_model.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": user_input,
    #         }
    #     ],
    #     model="unsloth/Llama-3.2-11B-Vision-Instruct",
    #     stream=True
    # )
    rag = Rag()
    return rag.run(user_input)
    
time_keeper.setup_lecture(120, 5)
time_keeper.time_display()



st.title("ALEX")
st.subheader("Augmented Lecture Explainer")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Assign avatar based on role
    avatar = "https://media.istockphoto.com/id/689364180/de/vektor/l%C3%A4chelnd-cartoon-gesichtssymbol-positive-menschen-emotion.jpg?s=612x612&w=0&k=20&c=qaqzmx64h626Flc2E6-BZEwL8z17U-F-RVCVcqJU8ZA=" if message["role"] == "user" else "https://i.pravatar.cc/300?img=2" 
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# Add a session state to track whether live selection is active
if "liveselection" not in st.session_state:
    st.session_state.liveselection = False

# Top bar with dropdown and slider button
col1, col2 = st.columns([4, 1])
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
            "Theory of Computation"
        ]
    )

    if st.session_state.liveselection:
        selected_lecture = st.selectbox(
            "Select a lecture:",
            [f"Lecture {i}" for i in range(1, 13)]
        )
    else:
        selected_lecture = None


# Toggle for live selection
with col2:
    live_lecture = st.toggle("Live Lecture", value=True)
    if live_lecture:
        st.session_state.liveselection = True
    else:
        st.session_state.liveselection = False

# Accept user input
if prompt := st.chat_input("Ask me anything."):
    # Add user message to chat history
    # Display user message in chat message container
    with st.chat_message("user", avatar="https://m.media-amazon.com/images/I/41aFi1uxvyL._AC_UF894,1000_QL80_.jpg"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant",  avatar="https://m.media-amazon.com/images/I/41CXC76XjTL._AC_UF894,1000_QL80_.jpg"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})