import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd

# Page setup
st.set_page_config(page_title="E9 Forum RAG", layout="wide")

# Title
st.title("E9 Forum Assistant")

# API Key input
if "api_key" not in st.session_state:
    api_key = st.text_input("OpenAI API Key:", type="password")
    if st.button("Set API Key"):
        st.session_state.api_key = api_key
        st.session_state.client = OpenAI(api_key=api_key)
        st.success("API Key set!")
        st.rerun()
    st.stop()

# Upload pre-computed files
if "index_loaded" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        index_file = st.file_uploader("Upload FAISS index file:", type=["index"])
    with col2:
        embeddings_file = st.file_uploader("Upload embeddings file:", type=["pkl", "pickle"])
    with col3:
        metadata_file = st.file_uploader("Upload metadata file (CSV or Pickle):", type=["csv", "pkl", "pickle"])
    
    if index_file and metadata_file:
        with st.spinner("Loading files..."):
            # Save and load FAISS index
            with open("index.faiss", "wb") as f:
                f.write(index_file.getbuffer())
            st.session_state.index = faiss.read_index("index.faiss")
            
            # Load metadata
            if metadata_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(metadata_file)
            else:
                with open("metadata.pkl", "wb") as f:
                    f.write(metadata_file.getbuffer())
                st.session_state.df = pd.read_pickle("metadata.pkl")
            
            # Load model
            st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            st.session_state.index_loaded = True
            st.success("Files loaded successfully!")
            st.rerun()
    st.stop()

# Query interface
query = st.text_input("Ask a question about BMW E9:")
if query:
    with st.spinner("Searching..."):
        # Encode query
        query_embedding = st.session_state.model.encode([query])
        
        # Search index
        k = 3  # Number of results
        distances, indices = st.session_state.index.search(query_embedding, k)
        
        # Get results
        context = ""
        for i, idx in enumerate(indices[0]):
            if idx < len(st.session_state.df):
                title = st.session_state.df.iloc[idx].get("thread_title", f"Thread {idx}")
                content = st.session_state.df.iloc[idx]["full_text"] if "full_text" in st.session_state.df.columns else st.session_state.df.iloc[idx].get("thread_all_posts", "No content")
                context += f"\nTHREAD {i+1}: {title}\n{content[:2000]}...\n\n"
        
        # Generate response
        prompt = f"""As a BMW E9 expert, answer this question using ONLY the forum thread information below:
        
{context}

QUESTION: {query}
ANSWER:"""
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        # Display answer
        st.markdown("### Answer")
        st.write(response.choices[0].message.content)
        
        # Show sources
        with st.expander("View Source Threads"):
            for i, idx in enumerate(indices[0]):
                if idx < len(st.session_state.df):
                    title = st.session_state.df.iloc[idx].get("thread_title", f"Thread {idx}")
                    st.subheader(f"Thread {i+1}: {title}")
                    st.text(f"Relevance score: {1/(1+distances[0][i]):.2f}")
                    with st.expander("View content"):
                        content = st.session_state.df.iloc[idx]["full_text"] if "full_text" in st.session_state.df.columns else st.session_state.df.iloc[idx].get("thread_all_posts", "No content")
                        st.write(content[:3000] + "..." if len(content) > 3000 else content)
