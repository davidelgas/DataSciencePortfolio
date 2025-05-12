
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import gdown
import requests
import json

st.set_page_config(page_title="BMW 3.0 Knowledge Base", layout="wide")

if st.sidebar.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.title("BMW 3.0 Knowledge Base")

@st.cache_resource
def download_files():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/index.faiss"):
        gdown.download("https://drive.google.com/uc?id=1FshTD4u9Ag2U09D5lcuTdMA1nfhfetvL", "data/index.faiss", quiet=False)
    if not os.path.exists("data/threads.pkl"):
        gdown.download("https://drive.google.com/uc?id=1buMFeYNyEQaIC9u4rWVpCU8rzbY7F0PV", "data/threads.pkl", quiet=False)
    index = faiss.read_index("data/index.faiss")
    with open("data/threads.pkl", "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, pd.DataFrame):
        df = raw
    else:
        df = pd.DataFrame([{"content": str(x)} for x in raw])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, df, model

if "api_key_set" not in st.session_state:
    with st.form("api"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        if st.form_submit_button("Submit") and openai_key:
            st.session_state.openai_key = openai_key
            st.session_state.api_key_set = True
            st.rerun()

elif "files_loaded" not in st.session_state:
    with st.spinner("Loading..."):
        index, df, model = download_files()
        st.session_state.index = index
        st.session_state.df = df
        st.session_state.model = model
        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
        st.session_state.files_loaded = True
        st.rerun()

else:
    query = st.text_input("Ask a question")
    if query:
        vec = st.session_state.model.encode([query])
        _, idxs = st.session_state.index.search(vec, 3)
        context = ""
        for i, idx in enumerate(idxs[0]):
            row = st.session_state.df.iloc[idx]
            content = row.get("content", "")
            context += f"THREAD {i+1}:
{content}

"
        prompt = f"{context}
QUESTION: {query}
ANSWER:"
        res = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        st.write(res.choices[0].message.content)
