import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import os
import pickle
import io

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="BMW 3.0 Knowledge Base - RAG vs Standard GPT", layout="wide")

# Reset button in sidebar (moved after page config)
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Title
st.title("BMW 3.0 Knowledge Base - RAG vs Standard GPT")
st.markdown("### Compare RAG-enhanced responses with standard GPT-3.5")

# Step 1: API Keys
if "api_key_set" not in st.session_state:
    st.header("Step 1: Enter API Key")
    with st.form("api_keys_form"):
        openai_key = st.text_input("OpenAI API Key:", type="password")
        submitted = st.form_submit_button("Submit Key")

        if submitted:
            if openai_key:
                st.session_state.openai_key = openai_key
                st.session_state.api_key_set = True
                st.success("API key saved!")
                st.rerun()
            else:
                st.error("OpenAI API key is required")

# Step 2: File Upload
elif "files_loaded" not in st.session_state:
    st.header("Step 2: Upload Required Files")

    st.info("Please upload the following files:")

    # File upload widgets
    faiss_file = st.file_uploader("Upload FAISS index file (index.faiss)", type=["faiss"])
    threads_file = st.file_uploader("Upload threads data file (threads.pkl)", type=["pkl"])

    if faiss_file and threads_file:
        with st.spinner("Loading BMW E9 forum knowledge base..."):
            try:
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)

                # Save uploaded files
                with open("data/index.faiss", "wb") as f:
                    f.write(faiss_file.getvalue())

                with open("data/threads.pkl", "wb") as f:
                    f.write(threads_file.getvalue())

                # Load data
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
                st.warning("Please check that the files are in the correct format")
    else:
        st.warning("Please upload all required files to continue")

# Step 3: Query Interface
else:
    # Add a "Go Back" button
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_loaded"]
        st.rerun()

    # Setup display options
    st.sidebar.title("Display Options")
    show_sources = st.sidebar.checkbox("Show sources", value=False)
    
    # Main query interface
    query = st.text_input("Ask a question about BMW E9")

    if query:
        with st.spinner("Processing your question..."):
            # Get RAG-enhanced response
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

            # Get RAG-enhanced answer
            rag_prompt = f"""As a BMW E9 expert, answer this question using ONLY the information provided from the E9 forum:

{context}

QUESTION: {query}
ANSWER:"""

            rag_response = st.session_state.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.2
            )

            rag_answer = rag_response.choices[0].message.content

            # Get standard GPT answer
            standard_prompt = f"""As a BMW E9 expert, answer this question based on your general knowledge:

QUESTION: {query}
ANSWER:"""

            standard_response = st.session_state.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": standard_prompt}],
                temperature=0.2
            )

            standard_answer = standard_response.choices[0].message.content

            # Display answers in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("RAG-Enhanced Response")
                st.markdown("*Using BMW E9 forum knowledge*")
                st.write(rag_answer)
                
                # Show sources if requested
                if show_sources:
                    st.subheader("Forum Sources")
                    for i, idx in enumerate(indices[0]):
                        if idx < len(st.session_state.df):
                            st.markdown(f"**Thread {i+1}:**")
                            st.text_area(f"Content {i+1}",
                                        str(st.session_state.df.iloc[idx].get("full_text",
                                                                          st.session_state.df.iloc[idx].get("content",
                                                                                                         "No content")))[:1000],
                                        height=150)
            
            with col2:
                st.header("Standard GPT Response")
                st.markdown("*Using GPT's general knowledge*")
                st.write(standard_answer)