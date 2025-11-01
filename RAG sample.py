# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    COMPLETE PDF-RAG APPLICATION                              ║
# ║                    WITH DETAILED STEP-BY-STEP COMMENTS                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                         1. IMPORT LIBRARIES                                 │
# │ Purpose: Import all required libraries for PDF processing, embeddings,     │
# │          vector storage, LLM integration, and memory management            │
# └─────────────────────────────────────────────────────────────────────────────┘
import os
import streamlit as st
from PyPDF2 import PdfReader                    # PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain.embeddings import OpenAIEmbeddings  # Convert text to vectors
from langchain.vectorstores import FAISS       # Vector database for similarity search
from langchain.chat_models import ChatOpenAI   # LLM for response generation
from langchain.memory import ConversationBufferWindowMemory  # Conversation memory
from langchain.chains import ConversationalRetrievalChain  # RAG chain with memory
from langchain.callbacks import get_openai_callback  # Cost tracking
import tempfile

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    2. CONFIGURATION SETTINGS                               │
# │ Purpose: Define all key parameters for chunking, embeddings, retrieval,    │
# │          and LLM generation. These are the "hyperparameters" of RAG        │
# └─────────────────────────────────────────────────────────────────────────────┘
class RAGConfig:
    # Text Chunking Parameters
    CHUNK_SIZE = 1000              # Size of each text chunk in characters
    CHUNK_OVERLAP = 200            # Overlap between chunks to preserve context
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI's best embedding model
    
    # Retrieval Parameters
    TOP_K = 5                      # Number of most relevant chunks to retrieve
    SIMILARITY_THRESHOLD = 0.7     # Minimum similarity score for relevance
    
    # LLM Parameters
    MODEL_NAME = "gpt-4"           # LLM model for response generation
    TEMPERATURE = 0.2              # Low temperature for factual, consistent responses
    MAX_TOKENS = 1500              # Maximum length of generated response
    
    # Memory Configuration
    MEMORY_WINDOW = 5              # Number of conversation turns to remember

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                      3. PDF TEXT EXTRACTION                                │
# │ Purpose: Extract text content from uploaded PDF files                      │
# │ Input: PDF file                                                            │
# │ Output: Raw text string                                                    │
# │ Challenges: Handle multi-column layouts, tables, special characters       │
# └─────────────────────────────────────────────────────────────────────────────┘
def extract_text_from_pdf(pdf_file):
    """
    Extract text from PDF file using PyPDF2
    
    Process:
    1. Create PDF reader object
    2. Iterate through all pages
    3. Extract text from each page
    4. Combine all text into single string
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)  # Create PDF reader
        
        # Process each page individually
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()  # Extract text from current page
            text += f"\n--- Page {page_num + 1} ---\n"  # Add page markers
            text += page_text
            
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    
    return text

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                       4. TEXT CHUNKING STRATEGY                            │
# │ Purpose: Split large text into smaller, manageable chunks                  │
# │ Why: LLMs have context limits, embeddings work better on focused content  │
# │ Strategy: Recursive splitting (paragraphs → sentences → words)            │
# │ Key Parameters: chunk_size, chunk_overlap, separators                     │
# └─────────────────────────────────────────────────────────────────────────────┘
def create_chunks(text):
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter
    
    Chunking Strategy:
    1. Try to split by paragraphs first (\n\n)
    2. If too large, split by sentences (\n)
    3. If still too large, split by words (" ")
    4. Last resort: split by characters
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,        # Maximum chunk size
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,  # Overlap to preserve context
        length_function=len,                     # Function to measure length
        separators=["\n\n", "\n", " ", ""]      # Priority order for splitting
    )
    
    chunks = text_splitter.split_text(text)
    
    # Add metadata to each chunk for tracking
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_metadata.append({
            "chunk_id": i,
            "chunk_size": len(chunk),
            "text": chunk
        })
    
    return chunks, chunk_metadata

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    5. EMBEDDING GENERATION                                 │
# │ Purpose: Convert text chunks into numerical vectors                        │
# │ How: Use OpenAI's embedding model to create dense vector representations  │
# │ Why: Enables semantic similarity search (meaning-based, not keyword)      │
# │ Output: Vector database ready for similarity search                       │
# └─────────────────────────────────────────────────────────────────────────────┘
def create_vector_store(chunks):
    """
    Create vector database from text chunks
    
    Process:
    1. Initialize OpenAI embeddings model
    2. Convert each chunk to vector representation
    3. Store vectors in FAISS database
    4. Enable similarity search functionality
    """
    try:
        # Initialize embedding model
        embeddings = OpenAIEmbeddings(
            model=RAGConfig.EMBEDDING_MODEL,    # Use specified embedding model
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create vector store from chunks
        # FAISS automatically generates embeddings for each chunk
        vectorstore = FAISS.from_texts(
            texts=chunks,           # Text chunks to embed
            embedding=embeddings,   # Embedding model to use
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
        )
        
        st.success(f"✅ Created vector database with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                      6. MEMORY MANAGEMENT                                  │
# │ Purpose: Maintain conversation history for multi-turn dialogues           │
# │ Type: ConversationBufferWindowMemory - keeps last N conversations        │
# │ Benefits: Handles follow-up questions, maintains context                  │
# │ Memory Key: "chat_history" - used in prompt construction                  │
# └─────────────────────────────────────────────────────────────────────────────┘
def initialize_memory():
    """
    Initialize conversation memory for multi-turn dialogue
    
    Memory Type: ConversationBufferWindowMemory
    - Keeps last K conversation turns
    - Automatically manages token usage
    - Maintains conversation context
    """
    memory = ConversationBufferWindowMemory(
        k=RAGConfig.MEMORY_WINDOW,      # Number of turns to remember
        memory_key="chat_history",      # Key used in prompts
        return_messages=True,           # Return as message objects
        output_key="answer"             # Key for storing AI responses
    )
    return memory

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    7. RAG CHAIN CONSTRUCTION                               │
# │ Purpose: Combine retrieval + generation with memory                       │
# │ Components: Retriever (vector search) + LLM + Memory                     │
# │ Flow: Query → Retrieve Context → Generate Response → Update Memory        │
# │ Chain Type: ConversationalRetrievalChain                                  │
# └─────────────────────────────────────────────────────────────────────────────┘
def create_rag_chain(vectorstore, memory):
    """
    Create the complete RAG chain with retrieval, generation, and memory
    
    Components:
    1. Retriever: Searches vector database for relevant chunks
    2. LLM: Generates responses based on retrieved context
    3. Memory: Maintains conversation history
    """
    try:
        # Initialize LLM with specific parameters
        llm = ChatOpenAI(
            model_name=RAGConfig.MODEL_NAME,    # GPT-4 for best quality
            temperature=RAGConfig.TEMPERATURE,  # Low temp for factual responses
            max_tokens=RAGConfig.MAX_TOKENS,    # Response length limit
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create retriever from vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",           # Use cosine similarity
            search_kwargs={
                "k": RAGConfig.TOP_K,          # Number of chunks to retrieve
                "fetch_k": RAGConfig.TOP_K * 2  # Fetch more, then filter
            }
        )
        
        # Create conversational RAG chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,                           # Language model
            retriever=retriever,               # Vector search retriever
            memory=memory,                     # Conversation memory
            return_source_documents=True,      # Return sources for citations
            verbose=True                       # Enable detailed logging
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                      8. QUERY PROCESSING                                   │
# │ Purpose: Process user questions and generate RAG responses                │
# │ Steps: Query → Embed → Search → Retrieve → Generate → Cite               │
# │ Includes: Cost tracking, source citations, error handling                 │
# └─────────────────────────────────────────────────────────────────────────────┘
def process_query(qa_chain, question):
    """
    Process user question through the RAG pipeline
    
    Pipeline:
    1. Embed user question
    2. Search vector database
    3. Retrieve relevant chunks
    4. Generate response using LLM
    5. Update conversation memory
    6. Return response with sources
    """
    try:
        # Track API costs using LangChain callback
        with get_openai_callback() as cb:
            # Process question through RAG chain
            result = qa_chain({
                "question": question,
                # chat_history is automatically handled by memory
            })
            
            # Extract response and sources
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Calculate costs
            cost_info = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            }
            
            return answer, source_docs, cost_info
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, [], {}

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     9. SOURCE CITATION DISPLAY                             │
# │ Purpose: Show which document chunks were used for the response            │
# │ Benefits: Transparency, fact-checking, source verification                │
# │ Format: Chunk preview + metadata + similarity scores                     │
# └─────────────────────────────────────────────────────────────────────────────┘
def display_sources(source_docs):
    """
    Display the source documents used for generating the response
    
    Information shown:
    - Chunk content preview
    - Source metadata
    - Relevance indicators
    """
    if source_docs:
        st.subheader("📚 Sources Used:")
        
        for i, doc in enumerate(source_docs):
            with st.expander(f"Source {i+1}"):
                # Show chunk content
                st.write("**Content:**")
                st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
                # Show metadata
                st.write("**Metadata:**")
                st.json(doc.metadata)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                      10. STREAMLIT USER INTERFACE                          │
# │ Purpose: Create web interface for PDF upload and Q&A interaction         │
# │ Features: File upload, processing status, chat interface, cost tracking  │
# │ Session State: Maintains RAG chain and memory across interactions        │
# └─────────────────────────────────────────────────────────────────────────────┘
def main():
    """
    Main Streamlit application
    
    UI Components:
    1. Header and configuration
    2. PDF upload section
    3. Document processing
    4. Q&A interface
    5. Cost and performance tracking
    """
    # Page configuration
    st.set_page_config(
        page_title="PDF-RAG Application",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF-RAG Application with Detailed Architecture")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                    SIDEBAR CONFIGURATION                              │
    # │ Purpose: Display current settings and allow parameter adjustment      │
    # └───────────────────────────────────────────────────────────────────────┘
    st.sidebar.header("⚙️ RAG Configuration")
    st.sidebar.markdown(f"""
    **Current Settings:**
    - **Chunk Size:** {RAGConfig.CHUNK_SIZE} characters
    - **Chunk Overlap:** {RAGConfig.CHUNK_OVERLAP} characters  
    - **Embedding Model:** {RAGConfig.EMBEDDING_MODEL}
    - **LLM Model:** {RAGConfig.MODEL_NAME}
    - **Temperature:** {RAGConfig.TEMPERATURE}
    - **Top-K Retrieval:** {RAGConfig.TOP_K}
    - **Memory Window:** {RAGConfig.MEMORY_WINDOW} turns
    """)
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                      API KEY VALIDATION                               │
    # │ Purpose: Ensure OpenAI API key is available                          │
    # └───────────────────────────────────────────────────────────────────────┘
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 Please set your OPENAI_API_KEY environment variable")
        st.stop()
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                    SESSION STATE INITIALIZATION                       │
    # │ Purpose: Maintain RAG components across Streamlit reruns             │
    # └───────────────────────────────────────────────────────────────────────┘
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                       PDF UPLOAD SECTION                             │
    # │ Purpose: Handle PDF file upload and validation                       │
    # └───────────────────────────────────────────────────────────────────────┘
    st.header("1️⃣ Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to create a searchable knowledge base"
    )
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                    DOCUMENT PROCESSING PIPELINE                       │
    # │ Purpose: Process uploaded PDF through complete RAG pipeline          │
    # │ Steps: Extract → Chunk → Embed → Store → Initialize Chain           │
    # └───────────────────────────────────────────────────────────────────────┘
    if uploaded_file is not None:
        st.header("2️⃣ Document Processing")
        
        with st.spinner("Processing PDF... This may take a few moments."):
            # Step 1: Extract text from PDF
            with st.expander("📄 Text Extraction", expanded=False):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.success(f"✅ Extracted {len(text)} characters from PDF")
                    st.text_area("Extracted Text Preview", text[:500] + "...", height=100)
                else:
                    st.error("❌ Failed to extract text from PDF")
                    st.stop()
            
            # Step 2: Create chunks
            with st.expander("✂️ Text Chunking", expanded=False):
                chunks, chunk_metadata = create_chunks(text)
                st.success(f"✅ Created {len(chunks)} text chunks")
                
                # Display chunking statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", len(chunks))
                with col2:
                    avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
                    st.metric("Avg Chunk Size", f"{avg_size:.0f} chars")
                with col3:
                    st.metric("Overlap Size", f"{RAGConfig.CHUNK_OVERLAP} chars")
                
                # Show sample chunks
                if st.checkbox("Show sample chunks"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(chunk[:200] + "...")
            
            # Step 3: Create vector store
            with st.expander("🧠 Vector Store Creation", expanded=False):
                vectorstore = create_vector_store(chunks)
                if vectorstore:
                    st.success("✅ Vector database created successfully")
                else:
                    st.error("❌ Failed to create vector database")
                    st.stop()
            
            # Step 4: Initialize memory and RAG chain
            with st.expander("🔗 RAG Chain Initialization", expanded=False):
                st.session_state.memory = initialize_memory()
                st.session_state.qa_chain = create_rag_chain(vectorstore, st.session_state.memory)
                
                if st.session_state.qa_chain:
                    st.success("✅ RAG chain initialized successfully")
                else:
                    st.error("❌ Failed to initialize RAG chain")
                    st.stop()
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                      Q&A INTERFACE SECTION                           │
    # │ Purpose: Handle user questions and display responses                  │
    # │ Features: Question input, response display, source citations         │
    # └───────────────────────────────────────────────────────────────────────┘
    if st.session_state.qa_chain is not None:
        st.header("3️⃣ Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            help="Ask any question about the uploaded PDF content"
        )
        
        # Process question
        if st.button("🔍 Get Answer", type="primary"):
            if question:
                with st.spinner("Searching document and generating answer..."):
                    answer, sources, cost_info = process_query(st.session_state.qa_chain, question)
                    
                    if answer:
                        # Display answer
                        st.subheader("💬 Answer:")
                        st.write(answer)
                        
                        # Display sources
                        display_sources(sources)
                        
                        # Update total cost
                        st.session_state.total_cost += cost_info.get("total_cost", 0)
                        
                        # Display cost information
                        with st.expander("💰 Cost Information"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Tokens", cost_info.get("total_tokens", 0))
                            with col2:
                                st.metric("Prompt Tokens", cost_info.get("prompt_tokens", 0))
                            with col3:
                                st.metric("Completion Tokens", cost_info.get("completion_tokens", 0))
                            with col4:
                                st.metric("Query Cost", f"${cost_info.get('total_cost', 0):.4f}")
                            
                            st.metric("Session Total Cost", f"${st.session_state.total_cost:.4f}")
            else:
                st.warning("Please enter a question")
    
    # ┌───────────────────────────────────────────────────────────────────────┐
    # │                      ARCHITECTURE INFORMATION                         │
    # │ Purpose: Educational section showing RAG components                   │
    # └───────────────────────────────────────────────────────────────────────┘
    st.header("4️⃣ RAG Architecture Overview")
    
    with st.expander("🏗️ System Architecture", expanded=False):
        st.markdown("""
        ### RAG Pipeline Components:
        
        1. **📄 Document Input**: PDF file upload and validation
        2. **🔧 Text Extraction**: PyPDF2 extracts text from PDF pages
        3. **✂️ Text Chunking**: RecursiveCharacterTextSplitter creates overlapping chunks
        4. **🧠 Embedding Generation**: OpenAI text-embedding-3-large converts text to vectors
        5. **🗄️ Vector Storage**: FAISS stores and indexes embeddings for fast search
        6. **❓ Query Processing**: User question converted to embedding
        7. **🎯 Similarity Search**: Find most relevant chunks using cosine similarity
        8. **🧠 Memory Management**: ConversationBufferWindowMemory maintains context
        9. **📝 Prompt Construction**: Combine query, context, and memory
        10. **🤖 Response Generation**: GPT-4 generates answer based on retrieved context
        11. **📚 Source Citation**: Display chunks used for transparency
        """)
    
    with st.expander("⚙️ Technical Parameters", expanded=False):
        st.markdown(f"""
        ### Current Configuration:
        
        **Text Processing:**
        - Chunk Size: {RAGConfig.CHUNK_SIZE} characters
        - Chunk Overlap: {RAGConfig.CHUNK_OVERLAP} characters
        - Splitting Strategy: Recursive (paragraphs → sentences → words)
        
        **Embeddings:**
        - Model: {RAGConfig.EMBEDDING_MODEL}
        - Dimensions: 3072 (for text-embedding-3-large)
        - Cost: ~$0.00013 per 1K tokens
        
        **Vector Search:**
        - Database: FAISS (Facebook AI Similarity Search)
        - Search Type: Cosine similarity
        - Top-K Retrieval: {RAGConfig.TOP_K} chunks
        - Similarity Threshold: {RAGConfig.SIMILARITY_THRESHOLD}
        
        **Language Model:**
        - Model: {RAGConfig.MODEL_NAME}
        - Temperature: {RAGConfig.TEMPERATURE} (low for factual responses)
        - Max Tokens: {RAGConfig.MAX_TOKENS}
        - Context Window: ~128K tokens
        
        **Memory:**
        - Type: ConversationBufferWindowMemory
        - Window Size: {RAGConfig.MEMORY_WINDOW} conversation turns
        - Purpose: Handle follow-up questions and maintain context
        """)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                           APPLICATION ENTRY POINT                          │
# │ Purpose: Run the Streamlit application                                     │
# │ Usage: streamlit run app.py                                               │
# └─────────────────────────────────────────────────────────────────────────────┘
if __name__ == "__main__":
    main()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                             SETUP INSTRUCTIONS                              ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║ 1. Install required packages:                                               ║
# ║    pip install streamlit langchain openai PyPDF2 faiss-cpu                  ║
# ║                                                                              ║
# ║ 2. Set OpenAI API key:                                                      ║
# ║    export OPENAI_API_KEY="your-api-key-here"                               ║
# ║                                                                              ║
# ║ 3. Run the application:                                                     ║
# ║    streamlit run pdf_rag_app.py                                            ║
# ║                                                                              ║
# ║ 4. Upload a PDF and start asking questions!                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         DETAILED FLOW EXPLANATION                           ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║ 1. USER UPLOADS PDF                                                         ║
# ║    └── File validation and temporary storage                                ║
# ║                                                                              ║
# ║ 2. TEXT EXTRACTION                                                          ║
# ║    └── PyPDF2 reads each page and extracts text content                    ║
# ║                                                                              ║
# ║ 3. TEXT CHUNKING                                                           ║
# ║    └── RecursiveCharacterTextSplitter creates overlapping chunks           ║
# ║    └── Preserves context with 200-character overlap                        ║
# ║                                                                              ║
# ║ 4. EMBEDDING GENERATION                                                     ║
# ║    └── OpenAI text-embedding-3-large converts chunks to 3072-dim vectors   ║
# ║                                                                              ║
# ║ 5. VECTOR STORAGE                                                           ║
# ║    └── FAISS creates searchable index of embeddings                        ║
# ║                                                                              ║
# ║ 6. USER ASKS QUESTION                                                       ║
# ║    └── Question converted to embedding using same model                     ║
# ║                                                                              ║
# ║ 7. SIMILARITY SEARCH                                                        ║
# ║    └── FAISS finds top-5 most similar chunks using cosine similarity       ║
# ║                                                                              ║
# ║ 8. CONTEXT PREPARATION                                                      ║
# ║    └── Retrieved chunks + conversation memory + user question              ║
# ║                                                                              ║
# ║ 9. RESPONSE GENERATION                                                      ║
# ║    └── GPT-4 generates answer based on retrieved context                   ║
# ║    └── Low temperature (0.2) ensures factual, grounded responses           ║
# ║                                                                              ║
# ║ 10. MEMORY UPDATE                                                           ║
# ║     └── Store Q&A pair for future context in follow-up questions           ║
# ║                                                                              ║
# ║ 11. RESPONSE DISPLAY                                                        ║
# ║     └── Show answer + source citations + cost information                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝