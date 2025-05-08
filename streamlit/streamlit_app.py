import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os

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

# Cache the RAG setup to avoid reloading on each interaction
@st.cache_resource
def load_rag_chain():
    # Load your vector database - adjust paths as needed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # If you have a pre-built FAISS index
    vectorstore = FAISS.load_local("path/to/your/faiss_index", embeddings)
    
    # Or create one from documents if needed
    # from langchain.document_loaders import DirectoryLoader
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # loader = DirectoryLoader("./your_data/", glob="*.txt")
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # splits = text_splitter.split_documents(documents)
    # vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Set up LLM - use OpenAI, local model or other providers
    # This example uses OpenAI - you'll need to set OPENAI_API_KEY in your environment
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

# Footer with branding
st.markdown("---")
st.markdown("© 2025 Your Brand Name | Powered by Your Custom RAG")
