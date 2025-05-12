
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import gdown

st.set_page_config(page_title="BMW 3.0 Knowledge Base", layout="wide")

if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.title("BMW 3.0 Knowledge Base")

@st.cache_resource
def download_files_from_gdrive():
    os.makedirs("data", exist_ok=True)
    faiss_file_id = "1FshTD4u9Ag2U09D5lcuTdMA1nfhfetvL"
    pkl_file_id = "1buMFeYNyEQaIC9u4rWVpCU8rzbY7F0PV"
    if not os.path.exists("data/index.faiss"):
        st.info("Downloading index from Google Drive...")
        faiss_url = f"https://drive.google.com/uc?id={faiss_file_id}"
        gdown.download(faiss_url, "data/index.faiss", quiet=False)
    if not os.path.exists("data/threads.pkl"):
        st.info("Downloading BMW E9 forum data from Google Drive...")
        pkl_url = f"https://drive.google.com/uc?id={pkl_file_id}"
        gdown.download(pkl_url, "data/threads.pkl", quiet=False)
    index = faiss.read_index("data/index.faiss")
    with open("data/threads.pkl", "rb") as f:
        thread_data = pickle.load(f)
    if isinstance(thread_data, pd.DataFrame):
        df = thread_data
    else:
        data = []
        for i, item in enumerate(thread_data):
            if hasattr(item, 'page_content'):
                data.append({"id": i, "content": item.page_content})
            elif isinstance(item, dict):
                data.append(item)
            else:
                data.append({"id": i, "content": str(item)})
        df = pd.DataFrame(data)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, df, model

# More code goes here (omitted for brevity in notebook block)
print("Streamlit code block completed...")
