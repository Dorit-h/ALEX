import streamlit as st

def main():
    # Set page configuration
    st.set_page_config(page_title="Lecture Selector", layout="wide")

    # Initialize session state for content list
    if "content_list" not in st.session_state:
        st.session_state.content_list = ["Box 1", "Box 2", "Box 3"]

    # Custom CSS for styling
    st.markdown(
        """
        <style>
            body {
                background-color: #6CA4DC;
                font-family: 'Times New Roman', sans-serif;
                margin: 0;
                padding: 0;
                border: 10px solid #0464BC;
                box-sizing: border-box;
            }
            .banner {
                background-color: #0464BC;
                padding: 20px;
                text-align: center;
                font-family: 'Times New Roman', sans-serif;
                color: white;
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .content-box {
                background-color: #F5F5F5;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                font-family: 'Times New Roman', Times, serif;
                color: #0464BC;
                white-space: pre-wrap; /* Preserves line breaks */
            }
            .scroll-container {
                max-height: 400px;
                overflow-y: auto;
            }
            .ask-button {
                background-color: #6CA4DC;
                color: white;
                font-family: 'Times New Roman', Times, serif;
                font-size: 16px;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .ask-button:hover {
                background-color: #0464BC;
            }
            .spacer {
                margin-bottom: 30px; /* Adds space between dropdowns and content boxes */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Banner at the top
    st.markdown('<div class="banner">VOLKER</div>', unsafe_allow_html=True)

    # Top bar with dropdown and slider button
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_lecture = st.selectbox(
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
    with col2:
        live_lecture = st.toggle("Live", value=1)

    # Additional dropdown if live toggle is off
    if not live_lecture:
        selected_offline_lecture = st.selectbox(
            "Select a lecture:",
            [f"Lecture {i}" for i in range(1, 13)]
        )

    # Spacer between dropdowns and content boxes
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Content box list
    st.markdown("### Content Boxes")
    with st.container():
        st.markdown("<div class='scroll-container'>", unsafe_allow_html=True)
        for content in st.session_state.content_list:
            st.markdown(
                f"<div class='content-box'>{content}</div>", unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Text input area
    user_input = st.text_area("Ask me anything", key="user_input")

    # Ask button
    ask_button_clicked = st.button("Ask", key="ask_button")
    if ask_button_clicked and user_input.strip():  # Check if input is not empty
        # Add the new content to the list
        st.session_state.content_list.append(f"You asked:\n{user_input.strip()}")
        # Clear the text area
        st.session_state["user_input"] = ""
        st.experimental_rerun()  # Refresh the app to display the new content

if __name__ == "__main__":
    main()
