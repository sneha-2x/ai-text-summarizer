# streamlit_app.py
import streamlit as st
from summarizer import summarize, read_pdf_streamlit

st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("📄 AI Text Summarizer")

# ---------------- Sidebar options ----------------
st.sidebar.title("⚙️ Settings")
input_type = st.sidebar.radio("Input type:", ["Text Input", "Upload TXT File(s)", "Upload PDF(s)"])
mode = st.sidebar.radio("Summarization Mode:", ["Abstractive", "Extractive"])
model_choice = st.sidebar.selectbox("Abstractive Model:", ["facebook/bart-large-cnn", "t5-small"])
min_length = st.sidebar.number_input("Min length", value=60)
max_length = st.sidebar.number_input("Max length", value=180)
sentences = st.sidebar.number_input("Num sentences (extractive)", value=4)

st.write("Upload text or PDF(s) from the sidebar to see the summary here.")

# ---------------- Input Handling ----------------
texts = []
file_names = []

if input_type == "Text Input":
    text = st.text_area("Enter your text here:", height=200)
    if text:
        texts.append(text)
        file_names.append("Text Input")

elif input_type == "Upload TXT File(s)":
    uploaded_files = st.file_uploader(
        "Upload TXT file(s)", type=["txt"], accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            text = f.read().decode("utf-8")
            texts.append(text)
            file_names.append(f.name)
            st.subheader(f"Preview: {f.name}")
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

elif input_type == "Upload PDF(s)":
    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            text = read_pdf_streamlit(f)
            texts.append(text)
            file_names.append(f.name)
            st.subheader(f"Preview: {f.name}")
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

# ---------------- Summarization ----------------
if texts and st.button("✨ Summarize"):
    for i, t in enumerate(texts):
        st.subheader(f"Summary: {file_names[i]}")
        summary = summarize(
            t,
            mode=mode.lower(),
            model=model_choice,
            min_length=min_length,
            max_length=max_length,
            sentences=sentences
        )
        st.write(summary)
        st.markdown("---")
