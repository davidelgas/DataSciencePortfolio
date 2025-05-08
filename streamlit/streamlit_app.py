import streamlit as st
import os
import shutil

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✨",
    layout="centered"
)

# Initialize key in session state if not already there
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("RAG Knowledge Assistant")

# API Key handling - prominently displayed at the top
if not st.session_state.api_key_entered:
    st.warning("⚠️ OpenAI API Key Required")
    api_key = st.text_input("Please enter your OpenAI API Key to continue:", type="password")
    if st.button("Submit API Key"):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_entered = True
            st.success("API key set successfully!")
            st.rerun()
        else:
            st.error("API key cannot be empty.")
    st.stop()  # Stop execution until API key is provided

# Now that API key is confirmed, continue with the app
st.success("API key is set. Your RAG system is ready to use.")

# Install required packages if not available
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Required packages not found. Please make sure you have the following packages installed:")
    st.code("pip install langchain langchain_community langchain_openai faiss-cpu sentence-transformers")
    st.stop()

# Create a sample document if needed
if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)
    with open("data/sample.txt", "w") as f:
        f.write("This is a sample document for your RAG system. Replace with your actual content.")
    st.info("Created a sample document. Add your own files to the 'data' folder.")

# Simple function to read text files directly
def read_text_files_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Create a document dictionary with content and metadata
                    documents.append({
                        "page_content": content,
                        "metadata": {"source": filename}
                    })
            except Exception as e:
                st.warning(f"Error reading {filename}: {e}")
    return documents

# Function to load or create RAG system
@st.cache_resource
def setup_rag_system():
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load documents directly without using DirectoryLoader
    documents = read_text_files_from_directory("data")
    if not documents:
        st.warning("No documents found. The system will use a sample document.")
        with open("data/sample.txt", "w") as f:
            sample_text = "This is a sample document for your RAG system. Replace with your actual content."
            f.write(sample_text)
        documents = [{"page_content": sample_text, "metadata": {"source": "sample.txt"}}]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Convert documents to the format expected by split_documents
    from langchain.schema import Document
    doc_objects = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
    
    splits = text_splitter.split_documents(doc_objects)
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Set up LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

# File uploader for adding documents
uploaded_files = st.file_uploader("Upload documents to your knowledge base:", accept_multiple_files=True, type=["txt", "md"])
if uploaded_files:
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} files.")
    # Clear cache to rebuild index with new files
    st.cache_resource.clear()
    st.rerun()

# Load the RAG system
with st.spinner("Setting up knowledge base..."):
    try:
        qa_chain = setup_rag_system()
        ready = True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        st.error("Please check the error message and ensure all dependencies are installed.")
        ready = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if ready:
    if prompt := st.chat_input("Ask me anything about the documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] if m["role"] == "user"]
                response = qa_chain({"question": prompt, "chat_history": chat_history})
                
                st.markdown(response["answer"])
                
                with st.expander("View Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(f"*{doc.metadata.get('source', 'Unknown')}*")
                        st.markdown(doc.page_content)
                        st.divider()
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Add reset button in sidebar
st.sidebar.title("Settings")
if st.sidebar.button("Reset API Key"):
    st.session_state.api_key_entered = False
    st.rerun()
