# -*- coding: utf-8 -*-
"""
RAG Streamlit Web Application
A web interface for PDF-based Question Answering using RAG
"""

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import tempfile
import pickle

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1a1a1a;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #1a1a1a;
    }
    .sidebar-content {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        chat_model = ChatHuggingFace(llm=llm)
        return chat_model
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process the PDF
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
        
        # Split documents
        with st.spinner("Splitting documents..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
        
        # Create embeddings and vector store
        with st.spinner("Creating embeddings and vector store..."):
            embeddings = initialize_embeddings()
            if embeddings is None:
                return None
            
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vector_store, len(docs), len(chunks)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def create_rag_chain(vector_store, chat_model):
    """Create RAG chain"""
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the question based on the provided context. 
            If you cannot find the answer in the context, please say so clearly.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"],
        )
        
        # Format documents function
        def format_docs(retrieved_docs):
            return "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create parallel chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        # Create main chain
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | chat_model | parser
        
        return main_chain
        
    except Exception as e:
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

def main():
    # Header
    st.markdown("<h1 class='main-header'>📚 RAG Document Q&A System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload a PDF file to start asking questions about its content"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Process PDF button
            if st.button("🔄 Process Document", type="primary"):
                chat_model = initialize_llm()
                if chat_model is None:
                    st.error("Failed to initialize language model. Please check your API configuration.")
                    return
                
                result = process_pdf(uploaded_file)
                if result is not None:
                    vector_store, num_docs, num_chunks = result
                    st.session_state.vector_store = vector_store
                    
                    # Create RAG chain
                    rag_chain = create_rag_chain(vector_store, chat_model)
                    st.session_state.rag_chain = rag_chain
                    
                    st.success("✅ Document processed successfully!")
                    st.info(f"📊 Document Stats:\n- Pages: {num_docs}\n- Chunks: {num_chunks}")
                else:
                    st.error("Failed to process document. Please try again.")
        
        # Document status
        st.header("📊 System Status")
        if st.session_state.vector_store is not None:
            st.success("✅ Document loaded and ready")
        else:
            st.warning("⚠️ No document loaded")
        
        if st.session_state.rag_chain is not None:
            st.success("✅ RAG system ready")
        else:
            st.warning("⚠️ RAG system not initialized")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about the document?",
            key="question_input"
        )
        
        # Ask button
        col_ask, col_example = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🚀 Ask Question", type="primary")
        
        with col_example:
            if st.button("💡 Example Questions"):
                st.info("""
                Example questions you can ask:
                - What is the main topic of this document?
                - Can you summarize the key findings?
                - What are the conclusions mentioned?
                - Tell me about [specific topic from your document]
                """)
        
        # Process question
        if ask_button and question:
            if st.session_state.rag_chain is None:
                st.error("Please upload and process a document first!")
            else:
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = st.session_state.rag_chain.invoke(question)
                        st.session_state.chat_history.append((question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
    
    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. **Upload PDF**: Click 'Browse files' in the sidebar
        2. **Process**: Click 'Process Document' to analyze your PDF
        3. **Ask Questions**: Type your questions in the input box
        4. **Get Answers**: The AI will answer based on your document
        
        **Tips:**
        - Be specific in your questions
        - The system works best with factual questions
        - You can ask follow-up questions
        """)
        
        st.header("🔧 Technical Details")
        st.markdown("""
        - **Model**: Mixtral-8x7B-Instruct
        - **Embeddings**: all-MiniLM-L6-v2
        - **Vector Store**: FAISS
        - **Chunk Size**: 1000 characters
        - **Retrieval**: Top 3 similar chunks
        """)

if __name__ == "__main__":
    main()