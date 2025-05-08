import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil

# Page configuration
st.set_page_config(
    page_title="Your Brand RAG Assistant",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for branding
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        color: #2E4057;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Add your logo/branding
st.image("https://your-logo-url.com/logo.png", width=150)
st.markdown("<h1 class='main-header'>Your Brand Knowledge Assistant</h1>", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a button to force recreate the index
if st.sidebar.button("Recreate Index"):
    if os.path.exists("index"):
        shutil.rmtree("index")
    st.sidebar.success("Index folder removed. A new index will be created on next load.")
    st.experimental_rerun()

# Cache the RAG setup to avoid reloading on each interaction
@st.cache_resource
def load_rag_chain():
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # ALWAYS CREATE A NEW INDEX - skip loading to avoid pickle errors
    # Create index directory if it doesn't exist
    os.makedirs("index", exist_ok=True)
    
    # Load documents from data folder
    try:
        loader = DirectoryLoader("data", glob="**/*.txt")
        documents = loader.load()
        st.sidebar.info(f"Loaded {len(documents)} documents from the data folder.")
    except Exception as e:
        # Create a sample document if no documents exist
        st.sidebar.warning("No documents found. Creating a sample document.")
        sample_text = """This is a sample document for your RAG system.
        Replace this with your actual data. You can add documents to the 'data' folder 
        in your repository, and they will be automatically loaded and indexed."""
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Write sample document
        with open("data/sample.txt", "w") as f:
            f.write(sample_text)
        
        # Load the sample document
        loader = TextLoader("data/sample.txt")
        documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    st.sidebar.info(f"Split into {len(splits)} chunks for indexing.")
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.sidebar.success("Vector store created successfully!")
    
    # Set up LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Create the conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

# Load the RAG chain with progress indicator
with st.spinner("Loading knowledge base..."):
    try:
        qa_chain = load_rag_chain()
        ready = True
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        st.error("Please check if your OpenAI API key is set correctly.")
        ready = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if ready:
    if prompt := st.chat_input("Ask me anything about our knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get chat history for context
                chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] 
                               if m["role"] == "user"]
                
                # Get response from RAG
                response = qa_chain({"question": prompt, "chat_history": chat_history})
                
                # Display the response
                st.markdown(response["answer"])
                
                # Optionally show sources
                with st.expander("View Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(f"*{doc.metadata.get('source', 'Unknown')}*")
                        st.markdown(doc.page_content)
                        st.divider()
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Add OpenAI API key input to sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API key set!")

# Add file uploader to sidebar for adding documents
st.sidebar.title("Add Documents")
uploaded_files = st.sidebar.file_uploader("Upload text files", accept_multiple_files=True, type=["txt", "md", "pdf"])
if uploaded_files:
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save uploaded files
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    
    st.sidebar.success(f"Uploaded {len(uploaded_files)} files. Click 'Recreate Index' to include them in your knowledge base.")

# Footer with branding
st.markdown("---")
st.markdown("© 2025 Your Brand Name | Powered by Your Custom RAG")
