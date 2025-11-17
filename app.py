# ============================================================================
# app.py - Smart Document Assistant with LangChain
# ============================================================================
# This is a complete Streamlit application that allows users to:
# 1. Upload PDF and text documents
# 2. Ask questions about those documents
# 3. Get AI-powered answers with source citations
# 4. Maintain conversation context (the AI remembers previous questions)
# ============================================================================

# ----------------------------------------------------------------------------
# IMPORTS - Loading all the tools and libraries we need
# ----------------------------------------------------------------------------

import streamlit as st  # Web framework for creating the user interface
import os  # For file and operating system operations
import tempfile  # For creating temporary files
from typing import List, Dict, Any  # For better code documentation and type hints

from langchain_community.document_loaders import PyPDFLoader, TextLoader  # For reading PDF and text files
import pypdf  # PDF processing library

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # For converting text into numerical vectors
from langchain.vectorstores import FAISS  # Database for storing and searching document vectors
from langchain.memory import ConversationBufferWindowMemory  # For remembering conversation history
from langchain.chains import ConversationalRetrievalChain  # For question-answering with context
from langchain_openai import ChatOpenAI  # OpenAI's GPT models
from langchain.callbacks import StreamlitCallbackHandler  # For displaying AI responses in real-time


# ============================================================================
# MAIN CLASS: SmartDocumentAssistant
# ============================================================================
# This class contains all the logic for processing documents and answering questions
# Think of it as the "brain" of our application
# ============================================================================

class SmartDocumentAssistant:
    """
    A comprehensive document assistant that provides context-aware responses
    based on uploaded documents with conversation memory.
    
    What this class does:
    - Processes uploaded documents (PDF, TXT)
    - Converts documents into searchable format
    - Answers questions based on document content
    - Remembers previous conversations for better context
    """
    
    def __init__(self):
        """
        Initialize the assistant - this runs when we first create the assistant.
        Sets up all the AI components we need.
        """
        
        # Initialize storage variables (will be filled in later)
        self.embeddings = None  # Will store the text-to-vector converter
        self.vectorstore = None  # Will store the searchable document database
        self.conversation_chain = None  # Will handle question-answering
        self.chat_history = []  # List to store conversation history
        
        os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
        
        if "OPENAI_API_KEY" not in os.environ:
            st.error("Please set your OpenAI API key in the .env file")
            st.stop()  # Stop the application if no API key
        
        # Initialize OpenAI Embeddings
        # Embeddings convert text into numbers (vectors) that computers can compare
        # This helps find relevant document sections for each question
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize the Language Model (LLM)
        # This is the AI that actually generates answers to questions
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Use GPT-3.5 (faster and cheaper than GPT-4)
            temperature=0.1,  # Low temperature = more focused, factual answers
            streaming=True  # Enable streaming for real-time response display
        )
        
        # Initialize conversation memory
        # This allows the AI to remember previous questions and answers
        # k=5 means it remembers the last 5 question-answer pairs
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",  # Key name for storing history
            return_messages=True,  # Return full message objects (not just strings)
            output_key="answer"  # Key name for the AI's answers
        )
    
    def process_documents(self, uploaded_files) -> bool:
        """
        Process uploaded documents and create vector embeddings for retrieval.
        
        This is a multi-step process:
        1. Load each uploaded file
        2. Extract text content
        3. Split text into smaller chunks
        4. Convert chunks into vectors (embeddings)
        5. Store vectors in a searchable database
        
        Args:
            uploaded_files: List of files uploaded by the user
            
        Returns:
            bool: True if successful, False if there was an error
        """
        try:
            # Step 1: Initialize list to store all document content
            documents = []
            
            # Step 2: Process each uploaded file
            for uploaded_file in uploaded_files:
                
                # Create a temporary file to store the uploaded content
                # We need to save it temporarily because document loaders need file paths
                with tempfile.NamedTemporaryFile(
                    delete=False,  # Don't delete immediately
                    suffix=f".{uploaded_file.name.split('.')[-1]}"  # Keep original file extension
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())  # Write uploaded content to temp file
                    tmp_file_path = tmp_file.name  # Save the file path
                
                # Choose the appropriate loader based on file type
                if uploaded_file.name.endswith('.pdf'):
                    # Use PDF loader for PDF files
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith('.txt'):
                    # Use text loader for TXT files
                    loader = TextLoader(tmp_file_path)
                else:
                    # If file type is not supported, show error and skip
                    st.error(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                # Load the document content
                file_documents = loader.load()
                
                # Add metadata to each document chunk
                # This helps us remember which file each piece of text came from
                for doc in file_documents:
                    doc.metadata['source_file'] = uploaded_file.name
                
                # Add all document chunks to our main list
                documents.extend(file_documents)
                
                # Clean up: delete the temporary file
                os.unlink(tmp_file_path)
            
            # Step 3: Check if we successfully loaded any documents
            if not documents:
                return False
            
            # Step 4: Split documents into smaller chunks
            # Why? Large documents are hard to search and process
            # Smaller chunks make it easier to find relevant information
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Each chunk will be ~1000 characters
                chunk_overlap=200,  # Overlap chunks by 200 chars to preserve context
                length_function=len  # Use character count for measuring length
            )
            
            # Perform the splitting
            split_documents = text_splitter.split_documents(documents)
            
            # Step 5: Create vector store (searchable database)
            # This converts all text chunks into vectors and stores them
            # FAISS is a fast similarity search library from Facebook
            self.vectorstore = FAISS.from_documents(
                documents=split_documents,  # The text chunks to store
                embedding=self.embeddings  # The embedding model to use
            )
            
            # Step 6: Create the conversational retrieval chain
            # This combines:
            # - Document retrieval (finding relevant chunks)
            # - Language model (generating answers)
            # - Memory (remembering conversation history)
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,  # The language model to use
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
                ),
                memory=self.memory,  # Conversation memory
                return_source_documents=True,  # Include source citations in response
                verbose=True  # Print debug information
            )
            
            # Success! Return True
            return True
            
        except Exception as e:
            # If anything goes wrong, show error message and return False
            st.error(f"Error processing documents: {str(e)}")
            return False
    
    def get_response(self, question: str, callback_handler) -> Dict[str, Any]:
        """
        Get context-aware response to user question.
        
        This method:
        1. Takes the user's question
        2. Searches for relevant document chunks
        3. Combines the chunks with conversation history
        4. Generates an answer using the AI
        5. Returns the answer with source citations
        
        Args:
            question: The user's question as a string
            callback_handler: Handler for streaming responses to Streamlit
            
        Returns:
            Dictionary containing:
            - "answer": The AI's response
            - "source_documents": List of relevant document chunks used
        """
        
        # Check if documents have been processed
        if not self.conversation_chain:
            return {
                "answer": "Please upload documents first.",
                "source_documents": []
            }
        
        try:
            # Call the conversation chain with the question
            # The chain will:
            # 1. Search for relevant document chunks
            # 2. Combine with conversation history
            # 3. Generate answer
            response = self.conversation_chain({
                "question": question,
                "callbacks": [callback_handler]  # For real-time streaming
            })
            
            return response
            
        except Exception as e:
            # If error occurs, return error message
            return {
                "answer": f"Error generating response: {str(e)}",
                "source_documents": []
            }
    
    def clear_memory(self):
        """
        Clear conversation memory.
        
        This resets the conversation history, making the AI "forget"
        previous questions and answers. Useful for starting fresh.
        """
        self.memory.clear()  # Clear the memory buffer
        self.chat_history = []  # Clear the chat history list


# ============================================================================
# MAIN FUNCTION: The Streamlit Application
# ============================================================================
# This function creates the user interface and handles user interactions
# ============================================================================

def main():
    """
    Main Streamlit application.
    
    This function:
    1. Sets up the page layout
    2. Creates the user interface
    3. Handles file uploads
    4. Manages the chat interface
    5. Displays results and statistics
    """
    
    # --------------------------------------------------------------------
    # PAGE CONFIGURATION
    # --------------------------------------------------------------------
    # Set up the basic page settings
    st.set_page_config(
        page_title="Smart Document Assistant",  # Browser tab title
        page_icon="📚",  # Browser tab icon
        layout="wide"  # Use full width of the browser
    )
    
    # --------------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # --------------------------------------------------------------------
    # Session state persists data between page reruns
    # This is how we remember information as the user interacts with the app
    
    # Initialize the assistant if it doesn't exist yet
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SmartDocumentAssistant()
    
    # Initialize chat history if it doesn't exist yet
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # --------------------------------------------------------------------
    # HEADER SECTION
    # --------------------------------------------------------------------
    # Display the main title and description
    st.title("📚 Smart Document Assistant")
    st.markdown("*Upload documents and ask questions with full context awareness and source citations*")
    
    # --------------------------------------------------------------------
    # SIDEBAR - Document Upload and Settings
    # --------------------------------------------------------------------
    # The sidebar is on the left side of the page
    with st.sidebar:
        
        # Document Upload Section
        st.header("📁 Document Upload")
        
        # File uploader widget
        # This creates a button that opens a file picker
        uploaded_files = st.file_uploader(
            "Choose files",  # Label
            accept_multiple_files=True,  # Allow multiple file selection
            type=['pdf', 'txt'],  # Only accept PDF and TXT files
            help="Upload PDF or text files to create your knowledge base"  # Tooltip
        )
        
        # If files are uploaded, show the "Process Documents" button
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                # Show a spinner while processing (visual feedback)
                with st.spinner("Processing documents..."):
                    # Process the documents
                    success = st.session_state.assistant.process_documents(uploaded_files)
                    
                    # Show success or error message
                    if success:
                        st.success(f"Successfully processed {len(uploaded_files)} documents!")
                        # Set a flag to indicate documents are ready
                        st.session_state.documents_processed = True
                    else:
                        st.error("Failed to process documents.")
        
        # Divider line for visual separation
        st.divider()
        
        # Memory Management Section
        st.header("🧠 Memory Management")
        
        # Button to clear conversation memory
        if st.button("Clear Conversation Memory"):
            st.session_state.assistant.clear_memory()
            st.session_state.chat_history = []
            st.success("Memory cleared!")
        
        # Another divider
        st.divider()
        
        # Application Info Section
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Upload** your documents (PDF/TXT)
        2. **Ask** questions about the content
        3. **Get** context-aware answers with sources
        4. **Continue** the conversation with memory
        """)
    
    # --------------------------------------------------------------------
    # MAIN CONTENT AREA - Chat Interface and Statistics
    # --------------------------------------------------------------------
    # Create two columns: left for chat, right for statistics
    col1, col2 = st.columns([2, 1])  # Left column is 2x wider than right
    
    # LEFT COLUMN - Chat Interface
    with col1:
        st.header("💬 Chat Interface")
        
        # Display all previous questions and answers
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            # Create an expandable section for each Q&A pair
            # Show first 50 characters of question in the header
            with st.expander(
                f"Q{i+1}: {question[:50]}...",
                expanded=(i == len(st.session_state.chat_history)-1)  # Expand only the latest
            ):
                # Display the question
                st.markdown(f"**Question:** {question}")
                
                # Display the answer
                st.markdown(f"**Answer:** {answer}")
                
                # Display source documents if available
                if sources:
                    st.markdown("**Sources:**")
                    # Show top 2 sources
                    for j, source in enumerate(sources[:2]):
                        # Extract filename and page number from metadata
                        filename = source.metadata.get('source_file', 'Unknown')
                        page = source.metadata.get('page', 'N/A')
                        st.markdown(f"- *{filename}* (Page {page})")
        
        # Question Input Section
        # Text input box for asking questions
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What are the main topics discussed in the documents?",
            key="question_input"  # Unique key for this widget
        )
        
        # Create two columns for buttons
        col_ask, col_clear = st.columns([1, 1])
        
        # "Ask Question" button
        with col_ask:
            ask_button = st.button("Ask Question", type="primary", use_container_width=True)
        
        # "Clear Chat" button
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()  # Refresh the page to show cleared chat
    
    # RIGHT COLUMN - Statistics and Recent Sources
    with col2:
        st.header("📊 Session Statistics")
        
        # Container for statistics
        stats_container = st.container()
        with stats_container:
            # Display number of documents processed
            st.metric(
                "Documents Processed",
                len(uploaded_files) if uploaded_files else 0
            )
            
            # Display number of questions asked
            st.metric(
                "Questions Asked",
                len(st.session_state.chat_history)
            )
            
            # Display memory buffer size
            memory_status = "5 exchanges" if hasattr(st.session_state.assistant, 'memory') else "Not initialized"
            st.metric("Memory Buffer", memory_status)
        
        # Recent Sources Section
        # Show sources from the most recent question
        if st.session_state.chat_history:
            st.header("📄 Recent Sources")
            
            # Get sources from the last question
            latest_sources = st.session_state.chat_history[-1][2] if st.session_state.chat_history else []
            
            # Display up to 3 sources
            for source in latest_sources[:3]:
                # Display filename
                st.markdown(f"- **{source.metadata.get('source_file', 'Unknown')}**")
                
                # Display page number
                st.markdown(f"  *Page {source.metadata.get('page', 'N/A')}*")
                
                # Show expandable preview of the content
                with st.expander("Preview"):
                    # Show first 200 characters
                    st.text(source.page_content[:200] + "...")
    
    # --------------------------------------------------------------------
    # PROCESS QUESTION - Handle "Ask Question" Button Click
    # --------------------------------------------------------------------
    # This runs when the user clicks "Ask Question"
    if ask_button and question:
        # Check if documents have been processed
        if not hasattr(st.session_state, 'documents_processed'):
            st.error("Please upload and process documents first!")
        else:
            # Show spinner while generating response
            with st.spinner("Generating response..."):
                # Create callback handler for streaming responses
                callback_handler = StreamlitCallbackHandler(st.container())
                
                # Get response from the assistant
                response = st.session_state.assistant.get_response(question, callback_handler)
                
                # Add question, answer, and sources to chat history
                st.session_state.chat_history.append((
                    question,  # The question
                    response["answer"],  # The AI's answer
                    response.get("source_documents", [])  # Source documents
                ))
                
                # Refresh the page to display the new Q&A
                st.rerun()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
# This is where the program starts running
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for the application.
    
    This block runs when you execute: streamlit run app.py
    """
    
    # Check if .env file exists
    # If not, create a template file
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        st.warning("Please add your OpenAI API key to the .env file and restart the application.")
    
    # Run the main application
    main()


# ============================================================================
# HOW TO USE THIS APPLICATION
# ============================================================================
# 
# 1. SETUP:
#    - Install required packages: pip install -r requirements.txt
#    - Create a .env file with your OpenAI API key
#    - Run: streamlit run app.py
#
# 2. UPLOAD DOCUMENTS:
#    - Click "Choose files" in the sidebar
#    - Select PDF or TXT files
#    - Click "Process Documents"
#
# 3. ASK QUESTIONS:
#    - Type your question in the text box
#    - Click "Ask Question"
#    - View the answer and source citations
#
# 4. CONTINUE CONVERSATION:
#    - Ask follow-up questions
#    - The AI remembers previous context
#    - Clear memory if you want to start fresh
#
# ============================================================================