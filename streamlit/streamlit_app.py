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

# Reset session state (to fix stuck state)
if 'reset_app' not in st.session_state:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.reset_app = True

# Page setup
st.set_page_config(page_title="E9 Forum Assistant", layout="wide")

# Title
st.title("E9 Forum Assistant")

# Simple architecture - no complex state management
openai_api_key = st.text_input("OpenAI API Key:", type="password")
serper_api_key = st.text_input("Serper API Key (optional):", type="password")

# Upload files only if API key is provided
if openai_api_key:
    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        faiss_file = st.file_uploader("Upload FAISS index file:", type=["index", "faiss"])
    with col2:
        thread_file = st.file_uploader("Upload thread data file:", type=["pkl", "pickle"])
    
    # Process button - separate from API key entry
    if faiss_file and thread_file and st.button("Process Files"):
        # Save and process files
        with st.spinner("Processing files..."):
            os.makedirs("data", exist_ok=True)
            
            # Save FAISS index
            with open("data/index.faiss", "wb") as f:
                f.write(faiss_file.getbuffer())
            
            # Save thread data
            with open("data/threads.pkl", "wb") as f:
                f.write(thread_file.getbuffer())
            
            # Load index
            index = faiss.read_index("data/index.faiss")
            
            # Load thread data
            with open("data/threads.pkl", "rb") as f:
                thread_data = pickle.load(f)
            
            # Convert to DataFrame if needed
            if isinstance(thread_data, pd.DataFrame):
                df = thread_data
            else:
                # Create simple DataFrame
                st.write("Converting data to DataFrame format...")
                try:
                    # Try a simple conversion
                    df = pd.DataFrame(thread_data)
                except:
                    # Fallback to basic structure
                    st.write("Using basic data structure...")
                    df = pd.DataFrame({
                        "thread_id": range(len(thread_data)),
                        "full_text": [str(item) for item in thread_data]
                    })
            
            # Load model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            st.session_state.index = index
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.has_web_search = bool(serper_api_key)
            st.session_state.serper_api_key = serper_api_key
            st.session_state.files_processed = True
            
            st.success("Files processed successfully!")
            st.rerun()
    
    # Query interface - only show after files are processed
    if st.session_state.get("files_processed", False):
        st.write("### Ask a question")
        
        # Search mode selection
        search_mode = "Forum Only"
        if st.session_state.get("has_web_search", False):
            search_mode = st.radio(
                "Search mode:",
                options=["Forum Only", "Web Only", "Forum + Web"]
            )
        
        # Question input
        query = st.text_input("Enter your question about BMW E9:")
        
        if query:
            with st.spinner("Searching..."):
                # Get forum results
                if search_mode in ["Forum Only", "Forum + Web"]:
                    # Encode query
                    query_embedding = st.session_state.model.encode([query])
                    
                    # Search index
                    distances, indices = st.session_state.index.search(query_embedding, 3)
                    
                    # Get results
                    context = ""
                    for i, idx in enumerate(indices[0]):
                        if idx < len(st.session_state.df):
                            content = str(st.session_state.df.iloc[idx].get("full_text", ""))
                            if len(content) > 1500:
                                content = content[:1500] + "..."
                            context += f"\nTHREAD {i+1}:\n{content}\n\n"
                    
                    # Generate response
                    prompt = f"""Answer using ONLY the forum information below:
                    
{context}

QUESTION: {query}
ANSWER:"""
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    
                    rag_answer = response.choices[0].message.content
                else:
                    rag_answer = ""
                
                # Get web results if needed
                if search_mode in ["Web Only", "Forum + Web"] and st.session_state.get("has_web_search", False):
                    try:
                        # Perform web search
                        url = "https://google.serper.dev/search"
                        payload = json.dumps({"q": f"BMW E9 {query}"})
                        headers = {
                            'X-API-KEY': st.session_state.serper_api_key,
                            'Content-Type': 'application/json'
                        }
                        response = requests.request("POST", url, headers=headers, data=payload)
                        search_results = response.json()
                        
                        # Format search results
                        web_context = "Web Search Results:\n\n"
                        if "organic" in search_results:
                            for i, result in enumerate(search_results["organic"][:3]):
                                title = result.get("title", "")
                                snippet = result.get("snippet", "")
                                web_context += f"Result {i+1}: {title}\n{snippet}\n\n"
                        
                        # Get web answer
                        web_prompt = f"""Using these web search results:
                        
{web_context}

Answer the question: {query}"""
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": web_prompt}],
                            temperature=0.2
                        )
                        
                        web_answer = response.choices[0].message.content
                    except Exception as e:
                        st.error(f"Web search error: {str(e)}")
                        web_answer = "Error retrieving web results."
                else:
                    web_answer = ""
                
                # Display answer
                st.markdown("### Answer")
                
                if search_mode == "Forum Only":
                    st.write(rag_answer)
                elif search_mode == "Web Only":
                    st.write(web_answer)
                else:  # Forum + Web
                    combined_prompt = f"""Combine these two answers:
                    
FORUM: {rag_answer}

WEB: {web_answer}

COMBINED ANSWER:"""
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": combined_prompt}],
                        temperature=0.2
                    )
                    
                    st.write(response.choices[0].message.content)
else:
    st.info("Please enter your OpenAI API key to continue.")
