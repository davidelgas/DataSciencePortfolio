
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import gdown
import requests
import json

st.set_page_config(page_title="BMW 3.0 Knowledge Base", layout="wide")

# Sidebar button to reset the application state
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

elif "files_loaded" not in st.session_state:
    st.header("Step 2: Loading BMW E9 Forum Data")
    with st.spinner("Loading BMW E9 forum knowledge base..."):
        try:
            index, df, model = download_files_from_gdrive()
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
            st.warning("Please check that files are properly shared on Google Drive and that the file IDs are correct.")

else:
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_loaded"]
        st.rerun()

    st.sidebar.title("Search Options")
    has_web_search = bool(st.session_state.get("serper_key", ""))
    search_options = ["Forum Only"]
    if has_web_search:
        search_options.extend(["Web Only", "Forum + Web"])
    search_mode = st.sidebar.radio("Search mode:", options=search_options)
    show_sources = st.sidebar.checkbox("Show sources", value=False)

    query = st.text_input("Search the BMW 3.0 Knowledge Base")

    if query:
        with st.spinner("Searching..."):
            forum_answer = ""
            web_answer = ""
            context = ""
            web_context = ""

            if search_mode in ["Forum Only", "Forum + Web"]:
                query_embedding = st.session_state.model.encode([query])
                k = 3
                distances, indices = st.session_state.index.search(query_embedding, k)
                retrieved_docs = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(st.session_state.df):
                        row = st.session_state.df.iloc[idx]
                        content_text = ""
                        for col in ["full_text", "content", "text", "page_content"]:
                            if col in row and pd.notna(row[col]):
                                content_text = str(row[col])
                                break
                        if not content_text:
                            content_text = str(row.to_dict())
                        if len(content_text) > 1500:
                            content_text = content_text[:1500] + "..."
                        retrieved_docs.append("FORUM THREAD {}:\n{}\n\n".format(i+1, content_text))
{content_text}

")
                context = "".join(retrieved_docs)

                if context:
                    forum_prompt = "As a BMW E9 expert, answer this question using ONLY the information provided from the E9 forum:\n\n{}\nQUESTION: {}\nANSWER:".format(context, query)

{context}
QUESTION: {query}
ANSWER:"""
                    try:
                        response = st.session_state.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": forum_prompt}],
                            temperature=0.2
                        )
                        forum_answer = response.choices[0].message.content
                    except Exception as e:
                        st.error(f"OpenAI API error (Forum Search): {str(e)}")
                        forum_answer = "Error getting answer from Forum context."
                else:
                    forum_answer = "No relevant information found in the forum for your query."

            if search_mode in ["Web Only", "Forum + Web"] and has_web_search:
                try:
                    search_query = f"BMW E9 {query}"
                    url = "https://google.serper.dev/search"
                    payload = json.dumps({"q": search_query})
                    headers = {
                        'X-API-KEY': st.session_state.serper_key,
                        'Content-Type': 'application/json'
                    }
                    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
                    response.raise_for_status()
                    search_results = response.json()

                    web_snippets = []
                    if "organic" in search_results:
                        for i, result in enumerate(search_results["organic"][:3]):
                            title = result.get("title", "N/A")
                            snippet = result.get("snippet", "N/A")
                            link = result.get("link", "#")
                            web_snippets.append(f"WEB RESULT {i+1}: Title: {title}
Snippet: {snippet}
Link: {link}

")
                    web_context = "".join(web_snippets)

                    if web_context:
                        web_prompt = "Using these web search results, answer the question about BMW E9:\n\n{}\nQUESTION: {}\nANSWER:".format(web_context, query)

{web_context}
QUESTION: {query}
ANSWER:"""
                        response = st.session_state.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": web_prompt}],
                            temperature=0.2
                        )
                        web_answer = response.choices[0].message.content
                    else:
                        web_answer = "No relevant information found from web search."

                except requests.exceptions.RequestException as e:
                    st.error(f"Web search request error: {str(e)}")
                    web_answer = "Error retrieving web results due to network issue."
                except Exception as e:
                    st.error(f"Web search processing error: {str(e)}")
                    web_answer = "Error processing web results."

            st.header("Answer")
            if search_mode == "Forum Only":
                st.write(forum_answer)
                source_type = "Forum"
            elif search_mode == "Web Only":
                st.write(web_answer)
                source_type = "Web"
            else:
                if forum_answer and web_answer and not forum_answer.startswith("Error") and not web_answer.startswith("Error"):
                    combined_prompt = "Combine these two answers to the question about BMW E9 \"{}\". Prioritize information from the forum if available, and supplement with web information. If there's conflicting information, state it.:\n\nFORUM ANSWER:\n{}\n\nWEB ANSWER:\n{}\n\nCOMBINED ANSWER:".format(query, forum_answer, web_answer)

FORUM ANSWER:
{forum_answer}

WEB ANSWER:
{web_answer}

COMBINED ANSWER:"""
                    try:
                        response = st.session_state.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": combined_prompt}],
                            temperature=0.2
                        )
                        combined_answer = response.choices[0].message.content
                        st.write(combined_answer)
                    except Exception as e:
                        st.error(f"OpenAI API error (Combining): {str(e)}")
                        st.write("Error combining answers. Displaying separately:")
                        st.subheader("Forum Answer:")
                        st.write(forum_answer)
                        st.subheader("Web Answer:")
                        st.write(web_answer)
                elif forum_answer and not forum_answer.startswith("Error"):
                    st.write(f"Displaying Forum Answer (Web search had issues or no results):

{forum_answer}")
                elif web_answer and not web_answer.startswith("Error"):
                    st.write(f"Displaying Web Answer (Forum search had issues or no results):

{web_answer}")
                else:
                    st.write("Could not retrieve a satisfactory answer from either forum or web search.")
                source_type = "Forum + Web"

            st.caption(f"Source: {source_type}")

            if show_sources:
                if search_mode in ["Forum Only", "Forum + Web"] and context:
                    with st.expander("View Forum Sources"):
                        st.markdown(context)
                if search_mode in ["Web Only", "Forum + Web"] and web_context:
                    with st.expander("View Web Sources"):
                        st.markdown(web_context)
