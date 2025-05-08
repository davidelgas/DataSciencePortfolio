import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import requests
import json
from urllib.parse import quote_plus

# Page setup
st.set_page_config(page_title="E9 Forum Assistant", layout="wide")

# Title
st.title("E9 Forum Assistant")

# Initialize session state
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

# API Key input
if "api_key" not in st.session_state:
    openai_api_key = st.text_input("OpenAI API Key:", type="password")
    serper_api_key = st.text_input("Serper API Key (for web search, optional):", type="password", 
                                 help="Get a free key at serper.dev")
    
    if st.button("Set API Keys"):
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.session_state.client = OpenAI(api_key=openai_api_key)
            
            if serper_api_key:
                st.session_state.serper_api_key = serper_api_key
                st.session_state.has_web_search = True
            else:
                st.session_state.has_web_search = False
                st.warning("No Serper API key provided. Web search will be unavailable.")
            
            st.success("API keys set!")
            st.rerun()
        else:
            st.error("OpenAI API key cannot be empty.")
    st.stop()

# Function to perform web search
def web_search(query, api_key):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

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
                elif isinstance(thread_data, list):
                    # It's a list of documents
                    # Create a DataFrame from the list
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

# Add options in sidebar
st.sidebar.title("Search Options")
search_mode = st.sidebar.radio(
    "Search mode:",
    options=["Forum Only", "Web Only", "Forum + Web"],
    index=0,
    disabled=not st.session_state.get("has_web_search", False) and (1 in [1, 2])
)

if not st.session_state.get("has_web_search", False) and search_mode in ["Web Only", "Forum + Web"]:
    st.sidebar.warning("Web search unavailable. Please add a Serper API key.")
    search_mode = "Forum Only"

st.sidebar.title("Display Options")
show_sources = st.sidebar.checkbox("Show sources", value=False)

# Query interface
st.write("### Ask a question")
query = st.text_input("Enter your question about BMW E9:")

if query:
    rag_answer = ""
    web_answer = ""
    
    # Forum search
    if search_mode in ["Forum Only", "Forum + Web"]:
        with st.spinner("Searching forum knowledge base..."):
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
            
            rag_answer = response.choices[0].message.content
    
    # Web search
    if search_mode in ["Web Only", "Forum + Web"] and st.session_state.get("has_web_search", False):
        with st.spinner("Searching the web..."):
            search_query = f"BMW E9 {query}"
            search_results = web_search(search_query, st.session_state.serper_api_key)
            
            # Format search results
            web_context = "Web Search Results:\n\n"
            
            # Add organic results
            if "organic" in search_results:
                for i, result in enumerate(search_results["organic"][:5]):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description")
                    link = result.get("link", "")
                    web_context += f"Result {i+1}: {title}\n{snippet}\nURL: {link}\n\n"
            
            # Generate response using web results
            web_prompt = f"""As a BMW E9 expert, answer this question using the web search results below.
            
{web_context}

QUESTION: {query}
ANSWER:"""
            
            response = st.session_state.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": web_prompt}],
                temperature=0.2
            )
            
            web_answer = response.choices[0].message.content
    
    # Display answer based on search mode
    st.markdown("### Answer")
    
    if search_mode == "Forum Only":
        st.write(rag_answer)
        source_label = "Forum Source Threads"
        answer_label = "From E9 Forum Knowledge Base"
    elif search_mode == "Web Only":
        st.write(web_answer)
        source_label = "Web Sources"
        answer_label = "From Web Search"
    else:  # Forum + Web
        # Combine answers
        combined_prompt = f"""I have two answers to the question "{query}" from different sources.

FORUM ANSWER: {rag_answer}

WEB ANSWER: {web_answer}

Please synthesize a single, comprehensive answer that combines unique information from both sources. 
If there are contradictions, note them clearly.

COMBINED ANSWER:"""
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.2
        )
        
        combined_answer = response.choices[0].message.content
        st.write(combined_answer)
        source_label = "Sources"
        answer_label = "Combined from Forum and Web"
    
    st.caption(answer_label)
    
    # Display sources if selected
    if show_sources:
        st.markdown(f"### {source_label}")
        
        # Show forum sources
        if search_mode in ["Forum Only", "Forum + Web"]:
            st.markdown("#### Forum Threads")
            for i, idx in enumerate(indices[0]):
                if idx < len(st.session_state.df):
                    st.markdown(f"**Thread {i+1}:** {st.session_state.df.iloc[idx].get('thread_title', f'Thread {idx}')}")
                    st.markdown(f"**Relevance score:** {1/(1+distances[0][i]):.2f}")
                    content = st.session_state.df.iloc[idx].get("full_text", "")
                    st.text_area(f"Content of Thread {i+1}", 
                                content[:2000] + "..." if len(content) > 2000 else content,
                                height=200)
                    st.markdown("---")
        
        # Show web sources
        if search_mode in ["Web Only", "Forum + Web"] and st.session_state.get("has_web_search", False):
            st.markdown("#### Web Results")
            if "organic" in search_results:
                for i, result in enumerate(search_results["organic"][:3]):
                    title = result.get("title", "No title")
                    link = result.get("link", "")
                    st.markdown(f"**{i+1}. [{title}]({link})**")
                    st.markdown(result.get("snippet", ""))
                    st.markdown("---")
