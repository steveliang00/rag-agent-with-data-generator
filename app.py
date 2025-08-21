import streamlit as st
import os
import glob
import shutil
from src.agent import Agent
from src.create_database import DocumentManager


def clear_data_folder():
    """Clear all PDF files from the data folder and ChromaDB on app startup"""
    try:
        # Create a temporary DocumentManager to get the paths
        temp_doc_manager = DocumentManager()
        data_path = temp_doc_manager.DATA_PATH
        chroma_path = temp_doc_manager.CHROMA_PATH
        
        # Find and remove all PDF files
        pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
        for pdf_file in pdf_files:
            os.remove(pdf_file)
            print(f"Removed: {pdf_file}")
        
        # Clear ChromaDB contents if folder exists
        if os.path.exists(chroma_path):
            # Remove all contents but keep the folder and .gitkeep
            for item in os.listdir(chroma_path):
                if item == '.gitkeep':
                    continue  # Preserve .gitkeep file
                item_path = os.path.join(chroma_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Cleared ChromaDB contents at: {chroma_path}")
        
        if pdf_files:
            print(f"Cleared {len(pdf_files)} PDF files from data folder")
            
    except Exception as e:
        print(f"Error clearing data folder: {e}")

# Clear PDFs from data folder on app startup
if 'app_initialized' not in st.session_state:
    clear_data_folder()
    st.session_state.app_initialized = True
    # Also clear the successful uploads tracking since we're starting fresh

    st.session_state.successful_uploads = []

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("PDF RAG Chatbot")

with st.sidebar:
    mistral_api_key = st.text_input("Enter your Mistral API Key", type="password")
    if mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = mistral_api_key

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    
    # Only process if this file hasn't been processed yet
    if file_name not in st.session_state.processed_files:
        try:
            # Save file to data directory using absolute path
            file_path = os.path.join(st.session_state.doc_manager.DATA_PATH, file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process and embed documents
            with st.spinner("Processing document..."):
                docs = st.session_state.doc_manager.load_docs()
                chunks = st.session_state.doc_manager.split_text(docs)
                st.session_state.doc_manager.embed_and_store_docs(chunks)
            
            # Mark file as processed
            st.session_state.processed_files.add(file_name)
            st.session_state.successful_uploads.append(file_name)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info(f"File {file_name} has already been processed.")

# Display all successful uploads
if st.session_state.successful_uploads:
    st.success("📁 **Successfully Processed Files:**")
    for i, filename in enumerate(st.session_state.successful_uploads, 1):
        st.success(f"✅ {i}. {filename}")
else:
    if uploaded_file is None:
        st.info("👆 Upload a PDF file above to get started!")

# Add a divider between upload and chat sections
if st.session_state.successful_uploads:
    st.divider()

if mistral_api_key:
    # Initialize the agent if not already done
    if st.session_state.agent is None:
        st.session_state.agent = Agent(api_provider="mistral")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Only show chat interface if we have processed documents
    if st.session_state.processed_files:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # Streamlit chat input
        user_input = st.chat_input("Type your question here...")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Get agent response
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run_agent(user_input)
                except Exception as e:
                    response = f"Error: {e}"

            # Add agent response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.info("Please upload a PDF file to start chatting!")
else:
    st.warning("Please enter your Mistral API key in the sidebar to use the chatbot.")




