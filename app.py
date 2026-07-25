import os

import streamlit as st

from ingest import ingest_pdf
from retrieval import ask_question


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")


# -----------------------------
# Upload Section
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    # Save PDF permanently
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Upload only once per session
    if "uploaded" not in st.session_state:

        with st.spinner("Uploading PDF to Pinecone..."):

            ingest_pdf(file_path)

        st.session_state.uploaded = True

        st.success("PDF uploaded successfully!")


# -----------------------------
# Chat Section
# -----------------------------

if st.session_state.get("uploaded"):

    st.divider()

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if st.button("Ask"):

        if question.strip():

            with st.spinner("Searching..."):

                answer = ask_question(question)

            st.subheader("Answer")

            st.write(answer)