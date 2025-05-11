import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import gdown
import time
import shutil

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="E9 Forum Assistant", layout="wide")

# Reset button in sidebar (moved after page config)
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Remove data directory entirely
    if os.path.exists("data"):
        shutil.rmtree("data")
    st.rerun()

# Add option to force redownload
force_download = st.sidebar.checkbox("Force redownload files", value=False)

# Title
st.title("E9 Forum Assistant")

# Function to download files from Google Drive
@st.cache_resource
def download_files_from_gdrive(force=False):
    """Download the FAISS index and thread data from Google Drive"""
    
    start_time = time.time()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Google Drive file IDs (your actual IDs)
    faiss_file_id = "1HIOd0eDy13RQOM5YXNhGSTsEyr_tUlks"
    pkl_file_id = "1pvCEfGz03j4Pt6wZMJKhGjDsKnGaSysW"
    
    download_status = {}
    
    # Download FAISS index
    faiss_path = "data/index.faiss"
    download_status['faiss_exists_before'] = os.path.exists(faiss_path)
    
    # Force download if requested
    if force and os.path.exists(faiss_path):
        os.remove(faiss_path)
        download_status['faiss_force_removed'] = True
    
    if not os.path.exists(faiss_path):
        st.info("Downloading FAISS index from Google Drive... (this may take a minute)")
        faiss_url = f"https://drive.google.com/uc?id={faiss_file_id}"
        download_start = time.time()
        
        # Use gdown with progress bar
        output = gdown.download(faiss_url, faiss_path, quiet=False)
        
        download_status['faiss_download_time'] = time.time() - download_start
        download_status['faiss_downloaded'] = True
        download_status['faiss_success'] = output is not None
    else:
        download_status['faiss_downloaded'] = False
        download_status['faiss_download_time'] = 0
    
    # Check FAISS file size
    if os.path.exists(faiss_path):
        download_status['faiss_size_mb'] = os.path.getsize(faiss_path) / (1024 * 1024)
    
    # Download thread data
    pkl_path = "data/threads.pkl"
    download_status['pkl_exists_before'] = os.path.exists(pkl_path)
    
    # Force download if requested
    if force and os.path.exists(pkl_path):
        os.remove(pkl_path)
        download_status['pkl_force_removed'] = True
    
    if not os.path.exists(pkl_path):
        st.info("Downloading thread data from Google Drive... (this may take a minute)")
        pkl_url = f"https://drive.google.com/uc?id={pkl_file_id}"
        download_start = time.time()
        
        # Use gdown with progress bar
        output = gdown.download(pkl_url, pkl_path, quiet=False)
        
        download_status['pkl_download_time'] = time.time() - download_start
        download_status['pkl_downloaded'] = True
        download_status['pkl_success'] = output is not None
    else:
        download_status['pkl_downloaded'] = False
        download_status['pkl_download_time'] = 0
    
    # Check PKL file size
    if os.path.exists(pkl_path):
        download_status['pkl_size_mb'] = os.path.getsize(pkl_path) / (1024 * 1024)
    
    # Load and return the data
    load_start = time.time()
    index = faiss.read_index(faiss_path)
    download_status['index_load_time'] = time.time() - load_start
    
    load_start = time.time()
    with open(pkl_path, "rb") as f:
        thread_data = pickle.load(f)
    download_status['pkl_load_time'] = time.time() - load_start
    
    # Convert to DataFrame if needed
    df_start = time.time()
    if isinstance(thread_data, pd.DataFrame):
        df = thread_data
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
        df = pd.DataFrame(data)
    download_status['df_conversion_time'] = time.time() - df_start
    download_status['num_threads'] = len(df)
    
    # Load sentence transformer model
    model_start = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    download_status['model_load_time'] = time.time() - model_start
    
    download_status['total_time'] = time.time() - start_time
    
    # Display diagnostic information
    st.expander("📊 Loading Diagnostics", expanded=True).write(f"""
    **File Status:**
    - FAISS index existed before: {download_status['faiss_exists_before']}
    - PKL file existed before: {download_status['pkl_exists_before']}
    {f"- FAISS file force removed: {download_status.get('faiss_force_removed', False)}" if force else ""}
    {f"- PKL file force removed: {download_status.get('pkl_force_removed', False)}" if force else ""}
    
    **Downloaded Files:**
    - FAISS downloaded: {download_status['faiss_downloaded']}
    - PKL downloaded: {download_status['pkl_downloaded']}
    {f"- FAISS download successful: {download_status.get('faiss_success', 'N/A')}" if download_status['faiss_downloaded'] else ""}
    {f"- PKL download successful: {download_status.get('pkl_success', 'N/A')}" if download_status['pkl_downloaded'] else ""}
    
    **File Sizes:**
    - FAISS: {download_status.get('faiss_size_mb', 0):.2f} MB
    - PKL: {download_status.get('pkl_size_mb', 0):.2f} MB
    
    **Timing (seconds):**
    - FAISS download: {download_status['faiss_download_time']:.2f}s
    - PKL download: {download_status['pkl_download_time']:.2f}s
    - Index loading: {download_status['index_load_time']:.2f}s
    - PKL loading: {download_status['pkl_load_time']:.2f}s
    - DataFrame conversion: {download_status['df_conversion_time']:.2f}s
    - Model loading: {download_status['model_load_time']:.2f}s
    - **Total time: {download_status['total_time']:.2f}s**
    
    **Data:**
    - Number of threads loaded: {download_status['num_threads']}
    
    **Google Drive URLs tested:**
    - FAISS: https://drive.google.com/uc?id={faiss_file_id}
    - PKL: https://drive.google.com/uc?id={pkl_file_id}
    """)
    
    return index, df, model

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

# Step 2: Auto-download and load files
elif "files_loaded" not in st.session_state:
    st.header("Step 2: Loading Files")
    
    # Show current force download setting
    if force_download:
        st.warning("Force download enabled - files will be downloaded fresh from Google Drive")
    
    with st.spinner("Loading files..."):
        try:
            # Download and load files (with diagnostics and force option)
            index, df, model = download_files_from_gdrive(force=force_download)
            
            # Save to session state
            st.session_state.index = index
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
            st.session_state.files_loaded = True
            
            st.success("Files loaded successfully!")
            st.info(f"Loaded {len(df)} forum threads")
            
            # Add a button to continue
            if st.button("Continue to Assistant"):
                st.rerun()
            
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.exception(e)  # This will show the full stack trace
            st.warning("Please make sure the Google Drive file IDs are correct")
            
            # Show manual upload option as fallback
            st.subheader("Manual Upload (Fallback)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Upload FAISS Index")
                faiss_file = st.file_uploader("Upload FAISS index file:", type=["index", "faiss"])
            
            with col2:
                st.subheader("Upload Thread Data")
                thread_file = st.file_uploader("Upload thread data file:", type=["pkl", "pickle"])
            
            if faiss_file and thread_file:
                if st.button("Process Uploaded Files"):
                    with st.spinner("Processing uploaded files..."):
                        try:
                            # Create directory if needed
                            os.makedirs("data", exist_ok=True)
                            
                            # Save FAISS index
                            with open("data/index.faiss", "wb") as f:
                                f.write(faiss_file.getbuffer())
                            
                            # Save thread data
                            with open("data/threads.pkl", "wb") as f:
                                f.write(thread_file.getbuffer())
                            
                            # Load the index
                            st.session_state.index = faiss.read_index("data/index.faiss")
                            
                            # Load thread data
                            with open("data/threads.pkl", "rb") as f:
                                thread_data = pickle.load(f)
                            
                            # Convert to DataFrame if needed
                            if isinstance(thread_data, pd.DataFrame):
                                st.session_state.df = thread_data
                            else:
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
                            st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
                            st.session_state.files_loaded = True
                            
                            st.success("Files processed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing uploaded files: {str(e)}")

# Step 3: Query Interface
else:
    # Add a "Go Back" button
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_loaded"]
        st.rerun()
    
    # Show info about current data
    st.sidebar.info(f"Loaded {len(st.session_state.df)} threads")
    
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
