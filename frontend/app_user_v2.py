import streamlit as st
import random
import time
from openai import OpenAI
import time_keeper


# Streamed response emulator
def response_generator(user_input: str):
    vlm_model = OpenAI(base_url="https://095kiew15yzv2e-8000.proxy.runpod.net/v1/", api_key="volker123")

    return vlm_model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="unsloth/Llama-3.2-11B-Vision-Instruct",
        stream=True
    )
    
time_keeper.setup_lecture(120, 5)
time_keeper.time_display()



st.title("ALEX")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
