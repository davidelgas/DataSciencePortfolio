import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle

# Force reset session state if needed
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Page setup
st.set_page_config(page_title="E9 Forum Assistant", layout="wide")

# Title
st.title("E9 Forum Assistant")

# Step 1: API Keys
if "api_key_set" not in st.session_state:
    st.header("Step 1: Enter API Keys")
    with st.form("api_keys_form"):
        openai_key = st.text_input("OpenAI API Key:", type="password")
        serper_key = st.text_input("Serper API Key (optional):", type="password")
        submitted = st.form_submit_button("Submit Keys")
        
        if submitted:
            if openai_key:
                st.session_state.openai_key = openai_key
                st.session_state.serper_key = serper_key
                st.session_state.api_key_set = True
                st.success("API keys saved!")
                st.rerun()
            else:
                st.error("OpenAI API key is required")

# Step 2: File Upload
elif "files_uploaded" not in st.session_state:
    st.header("Step 2: Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload FAISS Index")
        faiss_file = st.file_uploader("Upload FAISS index file:", type=["index", "faiss"])
    
    with col2:
        st.subheader("Upload Thread Data")
        thread_file = st.file_uploader("Upload thread data file:", type=["pkl", "pickle"])
    
    if faiss_file and thread_file:
        st.success("Files uploaded successfully!")
        
        with st.form("process_files_form"):
            st.write("Files are ready for processing")
            process_submitted = st.form_submit_button("Process Files")
            
            if process_submitted:
                with st.spinner("Processing files..."):
                    # Create directory if needed
                    os.makedirs("data", exist_ok=True)
                    
                    # Save FAISS index
                    with open("data/index.faiss", "wb") as f:
                        f.write(faiss_file.getbuffer())
                    
                    # Save thread data
                    with open("data/threads.pkl", "wb") as f:
                        f.write(thread_file.getbuffer())
                    
                    try:
                        # Load the index
                        st.session_state.index = faiss.read_index("data/index.faiss")
                        
                        # Load thread data
                        with open("data/threads.pkl", "rb") as f:
                            thread_data = pickle.load(f)
                        
                        # Convert to DataFrame if needed
                        if isinstance(thread_data, pd.DataFrame):
                            st.session_state.df = thread_data
                        else:
                            # Simple conversion
                            data = []
                            for i, item in enumerate(thread_data):
                                if hasattr(item, 'page_content'):
                                    data.append({"id": i, "content": item.page_content})
                                elif isinstance(item, dict):
                                    data.append(item)
                                else:
                                    data.append({"id": i, "content": str(item)})
                            
                            st.session_state.df = pd.DataFrame(data)
                        
                        # Load model
                        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
                        
                        st.session_state.files_uploaded = True
                        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
                        
                        st.success("Files processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")

# Step 3: Query Interface
else:
    # Add a "Go Back" button
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_uploaded"]
        st.rerun()
    
    # Setup search options
    st.sidebar.title("Search Options")
    has_web_search = bool(st.session_state.get("serper_key", ""))
    
    search_options = ["Forum Only"]
    if has_web_search:
        search_options.extend(["Web Only", "Forum + Web"])
    
    search_mode = st.sidebar.radio("Search mode:", options=search_options)
    show_sources = st.sidebar.checkbox("Show sources", value=False)
    
    # Main query interface
    st.header("Ask a Question")
    query = st.text_input("What would you like to know about BMW E9?")
    
    if query:
        with st.spinner("Searching..."):
            # Get forum results
            if search_mode in ["Forum Only", "Forum + Web"]:
                # Encode query
                query_embedding = st.session_state.model.encode([query])
                
                # Search index
                k = 3
                distances, indices = st.session_state.index.search(query_embedding, k)
                
                # Get forum context
                context = ""
                for i, idx in enumerate(indices[0]):
                    if idx < len(st.session_state.df):
                        row = st.session_state.df.iloc[idx]
                        # Try different column names
                        content = ""
                        for col in ["full_text", "content", "text", "page_content"]:
                            if col in row:
                                content = str(row[col])
                                break
                        
                        if not content:
                            # Just use the whole row as a string
                            content = str(row.to_dict())
                        
                        # Limit length
                        if len(content) > 1500:
                            content = content[:1500] + "..."
                        
                        context += f"\nTHREAD {i+1}:\n{content}\n\n"
                
                # Get forum answer
                forum_prompt = f"""As a BMW E9 expert, answer this question using ONLY the information provided:
                
{context}

QUESTION: {query}
ANSWER:"""
                
                response = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": forum_prompt}],
                    temperature=0.2
                )
                
                forum_answer = response.choices[0].message.content
            else:
                forum_answer = ""
                
            # Web search
            web_answer = ""
            if search_mode in ["Web Only", "Forum + Web"] and has_web_search:
                try:
                    import requests
                    import json
                    
                    # Simple web search
                    search_query = f"BMW E9 {query}"
                    url = "https://google.serper.dev/search"
                    
                    payload = json.dumps({"q": search_query})
                    headers = {
                        'X-API-KEY': st.session_state.serper_key,
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
                    web_prompt = f"""Using these web search results, answer the question:
                    
{web_context}

QUESTION: {query}
ANSWER:"""
                    
                    response = st.session_state.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": web_prompt}],
                        temperature=0.2
                    )
                    
                    web_answer = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Web search error: {str(e)}")
                    web_answer = "Error retrieving web results."
            
            # Display final answer
            st.header("Answer")
            
            if search_mode == "Forum Only":
                st.write(forum_answer)
                source_type = "Forum"
                final_answer = forum_answer
            elif search_mode == "Web Only":
                st.write(web_answer)
                source_type = "Web"
                final_answer = web_answer
            else:  # Forum + Web
                # Combine answers
                combined_prompt = f"""Combine these two answers to the question "{query}":
                
FORUM ANSWER: {forum_answer}

WEB ANSWER: {web_answer}

COMBINED ANSWER:"""
                
                response = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": combined_prompt}],
                    temperature=0.2
                )
                
                combined_answer = response.choices[0].message.content
                st.write(combined_answer)
                source_type = "Forum + Web"
                final_answer = combined_answer
            
            st.caption(f"Source: {source_type}")
            
            # Show sources if requested
            if show_sources and search_mode in ["Forum Only", "Forum + Web"]:
                st.subheader("Forum Sources")
                for i, idx in enumerate(indices[0]):
                    if idx < len(st.session_state.df):
                        st.markdown(f"**Thread {i+1}:**")
                        st.text_area(f"Content {i+1}", 
                                    str(st.session_state.df.iloc[idx].get("full_text", 
                                                                      st.session_state.df.iloc[idx].get("content", 
                                                                                                     "No content")))[:1000],
                                    height=150)
            
            if show_sources and search_mode in ["Web Only", "Forum + Web"] and has_web_search:
                st.subheader("Web Sources")
                if "organic" in search_results:
                    for i, result in enumerate(search_results["organic"][:3]):
                        st.markdown(f"**{result.get('title', f'Result {i+1}')}**")
                        st.markdown(f"*{result.get('snippet', 'No snippet')}*")
                        st.markdown(f"[Link]({result.get('link', '#')})")
                        st.markdown("---")
