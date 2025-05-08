import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle

# Page setup
st.set_page_config(page_title="E9 Forum RAG", layout="wide")

# Title
st.title("E9 Forum Assistant")

# Initialize session state
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

# API Key input
if "api_key" not in st.session_state:
    api_key = st.text_input("OpenAI API Key:", type="password")
    if st.button("Set API Key"):
        st.session_state.api_key = api_key
        st.session_state.client = OpenAI(api_key=api_key)
        st.success("API Key set!")
        st.rerun()
    st.stop()

# Upload necessary files
if not st.session_state.setup_complete:
    st.write("### Upload Files")
    st.write("Upload your FAISS index and CSV file to continue.")
    
    faiss_file = st.file_uploader("Upload FAISS index file:", type=["index", "faiss"])
    csv_file = st.file_uploader("Upload your forum CSV file:", type=["csv"])
    
    if st.button("Process Files") and faiss_file and csv_file:
        with st.spinner("Loading files..."):
            # Save and load FAISS index
            os.makedirs("data", exist_ok=True)
            with open("data/index.faiss", "wb") as f:
                f.write(faiss_file.getbuffer())
            
            # Save and load CSV
            with open("data/forum.csv", "wb") as f:
                f.write(csv_file.getbuffer())
            
            # Load the data
            st.session_state.index = faiss.read_index("data/index.faiss")
            st.session_state.df = pd.read_csv("data/forum.csv")
            
            # Create full_text column if it doesn't exist
            if "full_text" not in st.session_state.df.columns:
                st.session_state.df["full_text"] = (
                    st.session_state.df["thread_title"].fillna("") + "\n\n" +
                    st.session_state.df["thread_first_post"].fillna("") + "\n\n" +
                    st.session_state.df["thread_all_posts"].fillna("")
                )
            
            # Load model for encoding queries
            st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            st.session_state.setup_complete = True
            st.success("Files loaded successfully!")
            st.rerun()
    st.stop()

# Query interface
st.write("### Ask a question")
query = st.text_input("Enter your question about BMW E9:")

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
                title = st.session_state.df.iloc[idx]["thread_title"]
                content = st.session_state.df.iloc[idx]["full_text"]
                # Limit content length to avoid context issues
                if len(content) > 1500:
                    content = content[:1500] + "..."
                context += f"\nTHREAD {i+1}: {title}\n{content}\n\n"
        
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
        
        # Show sources (NO NESTED EXPANDERS - THIS FIXES THE ERROR)
        st.markdown("### Source Threads")
        
        for i, idx in enumerate(indices[0]):
            if idx < len(st.session_state.df):
                st.markdown(f"**Thread {i+1}:** {st.session_state.df.iloc[idx]['thread_title']}")
                st.markdown(f"**Relevance score:** {1/(1+distances[0][i]):.2f}")
                st.text_area(f"Content of Thread {i+1}", 
                            st.session_state.df.iloc[idx]["full_text"][:2000] + "..." 
                            if len(st.session_state.df.iloc[idx]["full_text"]) > 2000 
                            else st.session_state.df.iloc[idx]["full_text"],
                            height=200)
                st.markdown("---")
