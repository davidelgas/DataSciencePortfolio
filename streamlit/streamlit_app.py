import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="E9 Forum Assistant",
    page_icon="✨",
    layout="centered"
)

# Initialize key in session state if not already there
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize model and indexes in session state
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "df" not in st.session_state:
    st.session_state.df = None
if "ready" not in st.session_state:
    st.session_state.ready = False

# Title
st.title("E9 Forum Assistant")

# API Key handling - prominently displayed at the top
if not st.session_state.api_key_entered:
    st.warning("⚠️ OpenAI API Key Required")
    api_key = st.text_input("Please enter your OpenAI API Key to continue:", type="password")
    if st.button("Submit API Key"):
        if api_key:
            # Store API key in session state
            st.session_state.openai_api_key = api_key
            st.session_state.api_key_entered = True
            st.success("API key set successfully!")
            # Create OpenAI client
            st.session_state.client = OpenAI(api_key=api_key)
            st.rerun()
        else:
            st.error("API key cannot be empty.")
    st.stop()  # Stop execution until API key is provided

# Now that API key is confirmed, continue with the app
st.success("API key is set. Your RAG system is ready to use.")

# Create data directory if needed
if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

# Function to process forum thread CSV files
def process_forum_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Create the full_text column combining all thread info
    df["full_text"] = (
        df["thread_title"].fillna("") + "\n\n" +
        df["thread_first_post"].fillna("") + "\n\n" +
        df["thread_all_posts"].fillna("")
    )
    
    return df

# Function to setup semantic search model and index
def setup_semantic_search(df):
    with st.status("Setting up semantic search..."):
        st.write("Loading embedding model...")
        # Load SentenceTransformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        st.write("Encoding corpus... (this may take a while for large datasets)")
        # Create embeddings
        corpus_embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=False)
        
        st.write("Building FAISS index...")
        # Create FAISS index
        dimension = corpus_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(corpus_embeddings)
        
        return model, index

# File uploader for adding documents
uploaded_files = st.file_uploader("Upload your forum CSV file:", accept_multiple_files=False, type=["csv"])
if uploaded_files:
    with st.spinner("Processing uploaded file..."):
        # Save file
        with open(f"data/{uploaded_files.name}", "wb") as f:
            f.write(uploaded_files.getbuffer())
        
        # Process CSV
        df = process_forum_csv(f"data/{uploaded_files.name}")
        st.session_state.df = df
        st.success(f"Processed CSV with {len(df)} threads.")
        
        # Setup semantic search
        model, index = setup_semantic_search(df)
        st.session_state.model = model
        st.session_state.index = index
        st.session_state.ready = True
        
        st.success("Semantic search system is ready!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function for semantic search
def semantic_search(query, top_k=5):
    # Encode query
    query_embedding = st.session_state.model.encode([query])
    
    # Search
    distances, indices = st.session_state.index.search(query_embedding, top_k)
    
    retrieved_texts = []
    retrieved_info = []
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        title = st.session_state.df.iloc[idx]["thread_title"]
        thread_id = st.session_state.df.iloc[idx]["thread_id"]
        full_text = st.session_state.df.iloc[idx]["full_text"]
        
        retrieved_texts.append(full_text)
        retrieved_info.append({
            "title": title,
            "thread_id": thread_id,
            "distance": dist,
            "text": full_text
        })
    
    return retrieved_texts, retrieved_info

# Chat interface
if st.session_state.ready:
    if prompt := st.chat_input("Ask me anything about BMW E9 maintenance..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Perform semantic search
                retrieved_texts, retrieved_info = semantic_search(prompt, top_k=5)
                
                # Format context
                context = "\n\n".join([f"Thread {i+1}:\n{text}" for i, text in enumerate(retrieved_texts)])
                
                # Create prompt
                rag_prompt = f"""You are an expert on BMW E9 maintenance. Use the following forum threads to answer the question.

{context}

Question: {prompt}
Answer:"""
                
                # Get answer from OpenAI
                response = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": rag_prompt}],
                    temperature=0.2
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                
                # Show retrieved threads
                with st.expander("View Retrieved Threads"):
                    for i, info in enumerate(retrieved_info):
                        st.markdown(f"**Thread {i+1}:** {info['title']}")
                        st.markdown(f"**Distance:** {info['distance']:.4f}")
                        st.markdown(f"**Content Preview:**")
                        st.markdown(info['text'][:500] + "..." if len(info['text']) > 500 else info['text'])
                        st.divider()
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    if not uploaded_files:
        st.info("Please upload your forum thread CSV file to get started.")

# Add reset button in sidebar
st.sidebar.title("Settings")
if st.sidebar.button("Reset API Key"):
    st.session_state.api_key_entered = False
    st.rerun()

# Add advanced settings
st.sidebar.subheader("Search Settings")
top_k = st.sidebar.slider("Number of threads to retrieve", min_value=1, max_value=10, value=5)
