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
    st.write("Upload your FAISS index and thread pickle file to continue.")
    
    faiss_file = st.file_uploader("Upload FAISS index file:", type=["index", "faiss"])
    thread_file = st.file_uploader("Upload your thread pickle file:", type=["pkl", "pickle"])
    
    if st.button("Process Files") and faiss_file and thread_file:
        with st.spinner("Loading files..."):
            # Save and load FAISS index
            os.makedirs("data", exist_ok=True)
            with open("data/index.faiss", "wb") as f:
                f.write(faiss_file.getbuffer())
            
            # Save and load thread pickle
            with open("data/threads.pkl", "wb") as f:
                f.write(thread_file.getbuffer())
            
            # Load the data
            st.session_state.index = faiss.read_index("data/index.faiss")
            
            # Load thread data from pickle
            try:
                with open("data/threads.pkl", "rb") as f:
                    thread_data = pickle.load(f)
                
                # Check what's in the pickle file
                if isinstance(thread_data, pd.DataFrame):
                    # It's already a DataFrame
                    st.session_state.df = thread_data
                    st.write("Loaded thread data as DataFrame.")
                elif isinstance(thread_data, list):
                    # It's a list of documents
                    st.write(f"Loaded {len(thread_data)} thread documents.")
                    
                    # Create a DataFrame from the list
                    # Assuming each item has content and metadata
                    data = []
                    for i, doc in enumerate(thread_data):
                        item = {}
                        # Check possible formats
                        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                            # LangChain document format
                            item['thread_id'] = doc.metadata.get('thread_id', i)
                            item['thread_title'] = doc.metadata.get('thread_title', f'Thread {i}')
                            item['full_text'] = doc.page_content
                        elif isinstance(doc, dict):
                            # Dictionary format
                            item['thread_id'] = doc.get('thread_id', i)
                            item['thread_title'] = doc.get('thread_title', f'Thread {i}')
                            item['full_text'] = doc.get('content', doc.get('text', ''))
                        else:
                            # Assume it's just text
                            item['thread_id'] = i
                            item['thread_title'] = f'Thread {i}'
                            item['full_text'] = str(doc)
                            
                        data.append(item)
                    
                    st.session_state.df = pd.DataFrame(data)
                else:
                    # Unknown format
                    st.error(f"Unknown format in pickle file: {type(thread_data)}")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading thread data: {e}")
                st.stop()
            
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
                title = st.session_state.df.iloc[idx].get("thread_title", f"Thread {idx}")
                content = st.session_state.df.iloc[idx].get("full_text", "")
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
        
        # Show sources
        st.markdown("### Source Threads")
        
        for i, idx in enumerate(indices[0]):
            if idx < len(st.session_state.df):
                st.markdown(f"**Thread {i+1}:** {st.session_state.df.iloc[idx].get('thread_title', f'Thread {idx}')}")
                st.markdown(f"**Relevance score:** {1/(1+distances[0][i]):.2f}")
                content = st.session_state.df.iloc[idx].get("full_text", "")
                st.text_area(f"Content of Thread {i+1}", 
                            content[:2000] + "..." if len(content) > 2000 else content,
                            height=200)
                st.markdown("---")
