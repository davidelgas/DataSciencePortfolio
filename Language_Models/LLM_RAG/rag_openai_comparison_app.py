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
st.set_page_config(page_title="RAG vs OpenAI Comparison", layout="wide")

# Reset button in sidebar (moved after page config)
if st.sidebar.button("Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Title and description
st.title("RAG vs OpenAI Comparison")
st.markdown("""
This application allows you to compare responses between:
1. **RAG (Retrieval-Augmented Generation)**: Using your own knowledge base with OpenAI
2. **Pure OpenAI**: Directly using OpenAI's knowledge without your data
""")

# Step 1: API Keys
if "api_key_set" not in st.session_state:
    st.header("Step 1: Enter OpenAI API Key")
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
    st.header("Step 2: Upload Knowledge Base Files")

    st.info("Please upload the following files for your RAG system:")
    
    # File upload widgets
    faiss_file = st.file_uploader("Upload FAISS index file (index.faiss)", type=["faiss"])
    threads_file = st.file_uploader("Upload knowledge base data file (threads.pkl)", type=["pkl"])
    
    # Option to use sample data for demonstration
    use_sample = st.checkbox("I don't have these files, use sample data for demonstration")
    
    if use_sample or (faiss_file and threads_file):
        with st.spinner("Loading knowledge base..."):
            try:
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                if use_sample:
                    # For demo purposes, we'll create minimal sample data
                    # In a real app, you'd have pre-generated sample files
                    st.info("Using sample data for demonstration purposes. In a real application, upload your actual knowledge base files.")
                    
                    # Create a small sample index (2D, 3 entries)
                    sample_dimension = 384  # MiniLM-L6-v2 dimension
                    sample_index = faiss.IndexFlatL2(sample_dimension)
                    sample_vectors = np.random.random((3, sample_dimension)).astype('float32')
                    sample_index.add(sample_vectors)
                    
                    # Save sample index
                    faiss.write_index(sample_index, "data/index.faiss")
                    
                    # Create sample thread data
                    sample_threads = [
                        {"content": "This is a sample document about artificial intelligence. It discusses machine learning and neural networks."},
                        {"content": "Knowledge bases are collections of information that can be queried to find relevant data."},
                        {"content": "RAG systems combine retrieval from a knowledge base with language model generation."}
                    ]
                    
                    # Save sample threads
                    with open("data/threads.pkl", "wb") as f:
                        pickle.dump(sample_threads, f)
                    
                    index = sample_index
                    thread_data = sample_threads
                else:
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
                st.info(f"Loaded {len(df)} documents into the knowledge base")
                st.rerun()

            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
                st.warning("Please check that the files are in the correct format")
    else:
        st.warning("Please upload both required files or use sample data to continue")

# Step 3: Comparison Interface
else:
    # Add a "Go Back" button
    if st.sidebar.button("Go Back to File Upload"):
        del st.session_state["files_loaded"]
        st.rerun()

    # Main query interface
    st.header("Compare RAG and OpenAI Responses")
    
    # Sidebar options
    st.sidebar.title("Options")
    show_sources = st.sidebar.checkbox("Show RAG sources", value=True)
    model_choice = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Temperature slider
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # Number of retrieved documents
    k_docs = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)

    # Query input
    query = st.text_input("Enter your question")

    if query:
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with st.spinner("Generating responses..."):
            # Get RAG results
            with col1:
                st.subheader("RAG Response")
                
                # Encode query
                query_embedding = st.session_state.model.encode([query])

                # Search index
                distances, indices = st.session_state.index.search(query_embedding, k_docs)

                # Get context from knowledge base
                context = ""
                retrieved_docs = []
                
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

                        context += f"\nDOCUMENT {i+1}:\n{content}\n\n"
                        retrieved_docs.append({
                            "doc_id": i+1,
                            "content": content,
                            "similarity": float(distances[0][i])
                        })

                # Get RAG answer
                rag_prompt = f"""Answer this question using ONLY the information provided in the documents below:

{context}

QUESTION: {query}
ANSWER:"""

                response = st.session_state.client.chat.completions.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": rag_prompt}],
                    temperature=temperature
                )

                rag_answer = response.choices[0].message.content
                st.markdown(rag_answer)
                
                # Show sources if requested
                if show_sources:
                    st.subheader("Retrieved Documents")
                    for doc in retrieved_docs:
                        with st.expander(f"Document {doc['doc_id']} (Similarity: {doc['similarity']:.4f})"):
                            st.text_area(f"Content", doc['content'], height=150)

            # Get OpenAI direct results
            with col2:
                st.subheader("Pure OpenAI Response")
                
                direct_prompt = f"""Answer this question based on your knowledge:

QUESTION: {query}
ANSWER:"""

                response = st.session_state.client.chat.completions.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": direct_prompt}],
                    temperature=temperature
                )

                direct_answer = response.choices[0].message.content
                st.markdown(direct_answer)
            
            # Add a section to compare the answers
            st.header("Analysis of Differences")
            
            comparison_prompt = f"""Compare these two answers to the question "{query}" and analyze their differences:

RAG ANSWER (using retrieved documents): {rag_answer}

DIRECT ANSWER (using OpenAI's knowledge): {direct_answer}

List the key differences in information content, specificity, accuracy (if you can determine this), and any other notable distinctions:"""

            response = st.session_state.client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": comparison_prompt}],
                temperature=temperature
            )

            comparison = response.choices[0].message.content
            st.markdown(comparison)
