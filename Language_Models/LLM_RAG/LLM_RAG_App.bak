import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import gdown

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="BMW 3.0 Knowledge Base", layout="wide")

# Reset button in sidebar (moved after page config)
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Title
st.title("BMW 3.0 Knowledge Base")

# Function to download files from Google Drive
@st.cache_resource
def download_files_from_gdrive():
    """Download the FAISS index and thread data from Google Drive"""

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Google Drive file IDs (your actual IDs)
    faiss_file_id = "1TRrV9s50lA7qH45kCHJEvUxF_26awAns"
    pkl_file_id = "13x_LhBC-nuD64rqfI7JSYHUlUdlJsLEX"


    # Download FAISS index
    if not os.path.exists("data/index.faiss"):
        st.info("Downloading index from Google Drive...")
        faiss_url = f"https://drive.google.com/uc?id={faiss_file_id}"
        gdown.download(faiss_url, "data/index.faiss", quiet=False)

    # Download thread data
    if not os.path.exists("data/threads.pkl"):
        st.info("Downloading BMW E9 forum data from Google Drive...")
        pkl_url = f"https://drive.google.com/uc?id={pkl_file_id}"
        gdown.download(pkl_url, "data/threads.pkl", quiet=False)

    # Load and return the data
    index = faiss.read_index("data/index.faiss")

    with open("data/threads.pkl", "rb") as f:
        thread_data = pickle.load(f)

    # Convert to DataFrame if needed
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

    # Load sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
    st.header("Step 2: Loading BMW E9 Forum Data")

    with st.spinner("Loading BMW E9 forum knowledge base..."):
        try:
            # Download and load files
            index, df, model = download_files_from_gdrive()

            # Save to session state
            st.session_state.index = index
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
            st.session_state.files_loaded = True

            st.success("Knowledge base loaded successfully!")
            st.info(f"Loaded {len(df)} BMW E9 forum threads")
            st.rerun()

        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.warning("Please check that files are properly shared on Google Drive")

# Step 3: Query Interface
else:
    # Add a "Go Back" button
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_loaded"]
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
    query = st.text_input("Search the BMW 3.0 Knowledge Base")

    if query:
        with st.spinner("Searching BMW E9 knowledge base..."):
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

                        context += f"\nFORUM THREAD {i+1}:\n{content}\n\n"

                # Get forum answer
                forum_prompt = f"""As a BMW E9 expert, answer this question using ONLY the information provided from the E9 forum:

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
                    web_prompt = f"""Using these web search results, answer the question about BMW E9:

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
                combined_prompt = f"""Combine these two answers to the question about BMW E9 "{query}":

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
