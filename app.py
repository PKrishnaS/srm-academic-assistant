import streamlit as st
import os
import json
import time
import html
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from helper_functions import get_groq_llm, get_embeddings
from chain import create_qa_chain, create_checker_chain, check_answer_type_chain
from logger import setup_logger
import config

# Page configuration
st.set_page_config(
    page_title="SRM Academic Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize logger
logger = setup_logger()
print("✓ Logger initialized successfully")

# Initialize vectorstore and chains with aggressive caching
@st.cache_resource(show_spinner="Loading AI system from local cache...")
def init_vectorstore():
    """Initialize or load FAISS vectorstore - fully cached and persisted"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    faiss_index_path = "faiss_index"
    directory = config.DIRECTORY
    pdf_path = os.path.join(directory, 'data-1.pdf')  # SRMIST regulations
    
    index_faiss_file = os.path.join(faiss_index_path, 'index.faiss')
    index_pkl_file = os.path.join(faiss_index_path, 'index.pkl')
    
    # Priority 1: Load from existing index if both files exist
    if os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file):
        logger.info("Loading FAISS index from local disk (cached)...")
        try:
            vectorstore = FAISS.load_local(
                faiss_index_path, 
                get_embeddings(), 
                allow_dangerous_deserialization=True
            )
            logger.info("SUCCESS: Loaded from local cache - no rebuild needed!")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load cached index: {str(e)}")
            logger.info("Will rebuild index from PDF...")
    
    # Priority 2: Build new index only if cache doesn't exist or failed
    logger.info("Building new FAISS index from PDF...")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            f"Please place your SRMIST Academic Regulations PDF as 'data-1.pdf' in the '{directory}' folder."
        )
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,  # Use config value
        chunk_overlap=400,  # More overlap for better context
        length_function=len
    )
    
    # Load and split PDF
    logger.info("Loading PDF document...")
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    logger.info(f"Loaded {len(document)} pages from PDF")
    
    logger.info("Splitting into chunks...")
    chunks = text_splitter.split_documents(document)
    logger.info(f"Created {len(chunks)} chunks")
    
    if not chunks:
        raise ValueError("No content extracted from PDF. Please check if the PDF is valid.")
    
    # Create embeddings and vectorstore
    logger.info("Creating embeddings (this may take a few minutes)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk for future use
    logger.info(f"Saving vectorstore to {faiss_index_path}...")
    os.makedirs(faiss_index_path, exist_ok=True)
    vectorstore.save_local(faiss_index_path)
    logger.info("SUCCESS: Vectorstore created and saved to local disk!")
    logger.info(f"Location: {os.path.abspath(faiss_index_path)}")
    logger.info("Next server restart will load instantly from this cache!")
    
    return vectorstore

@st.cache_resource(show_spinner="Initializing AI chains...")
def init_chains(_vectorstore, _api_key=None, _model=None, _temperature=None):
    """Initialize LangChain chains - fully cached in memory"""
    logger.info("Creating AI chains (cached for session)...")
    llm = get_groq_llm(api_key=_api_key, model=_model, temperature=_temperature)
    qa_chain = create_qa_chain(llm, _vectorstore)
    checker_chain = create_checker_chain(llm)
    answer_type_chain = check_answer_type_chain(llm)
    logger.info("SUCCESS: AI chains initialized and cached!")
    return qa_chain, checker_chain, answer_type_chain

@st.cache_data
def load_embeddings_model():
    """Cache the embeddings model in memory to avoid reloading"""
    logger.info("Loading embeddings model (cached)...")
    return get_embeddings()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = []
    if 'show_metrics' not in st.session_state:
        st.session_state.show_metrics = False
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'custom_api_key' not in st.session_state:
        st.session_state.custom_api_key = config.GROQ_API_KEY
    if 'custom_model' not in st.session_state:
        st.session_state.custom_model = config.GROQ_MODEL
    if 'custom_temperature' not in st.session_state:
        st.session_state.custom_temperature = config.GROQ_TEMPERATURE
    if 'last_retrieved' not in st.session_state:
        st.session_state.last_retrieved = []

# Production-ready chatbot CSS - Classic & Clean
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide some Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Deploy button is visible - you'll see it in top-right when running */
    /* If you don't see it, it means you're running locally - that's normal! */
    /* The deploy button only appears after you deploy to Streamlit Cloud */
    
    /* Main Background */
    .stApp {
        background: #ffffff;
    }
    
    /* Container Spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 48rem !important;
    }
    
    /* Message Container */
    .message-row {
        display: flex;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-row.user {
        justify-content: flex-end;
    }
    
    .message-row.assistant {
        justify-content: flex-start;
    }
    
    /* Message Bubble */
    .message-bubble {
        max-width: 70%;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    
    .message-bubble.user {
        flex-direction: row-reverse;
    }
    
    /* Avatar */
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Message Content Box */
    .message-box {
        padding: 0.875rem 1.125rem;
        border-radius: 18px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .message-box.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-box.assistant {
        background: #f7f7f8;
        color: #202123;
        border-bottom-left-radius: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background: white;
        color: #353740;
        border: 1px solid #d9d9e3;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease;
    }
    
    .stButton > button:hover {
        background: #f7f7f8;
        border-color: #c5c5d2;
    }
    
    .stButton > button[kind="primary"] {
        background: #10a37f;
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #1a7f64;
    }
    
    /* Input Area */
    .stTextArea > div > div > textarea {
        background: white;
        border: 1px solid #d9d9e3;
        border-radius: 12px;
        color: #353740;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        resize: none;
        box-shadow: 0 0 0 0 rgba(16, 163, 127, 0);
        transition: all 0.15s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
        outline: none;
    }
    
    /* Header */
    .header-container {
        background: white;
        border-bottom: 1px solid #ececf1;
        padding: 0.75rem 0;
        margin-bottom: 0;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-title {
        font-size: 1rem;
        font-weight: 600;
        color: #353740;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 0.75rem;
        color: #8e8ea0;
        margin: 0;
        font-weight: 400;
    }
    
    /* Welcome State */
    .welcome-container {
        padding: 4rem 1rem;
        text-align: center;
        max-width: 40rem;
        margin: 0 auto;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        color: #202123;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
        color: #6e6e80;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Example Cards */
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.75rem;
        margin-top: 2rem;
    }
    
    .example-card {
        background: #f7f7f8;
        border: none;
        border-radius: 8px;
        padding: 1rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    
    .example-card:hover {
        background: #ececf1;
    }
    
    .example-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #202123;
        margin-bottom: 0.25rem;
    }
    
    /* Form Container */
    .input-container {
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #ececf1;
    }
    
    /* Status Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border: none;
        font-size: 0.875rem;
    }
    
    .stSuccess {
        background: #d1f4e0;
        color: #0f5132;
    }
    
    .stError {
        background: #fee;
        color: #c00;
    }
    
    .stInfo {
        background: #e7f3ff;
        color: #014361;
    }
    
    /* Metrics */
    .stMetric {
        background: #f7f7f8;
        padding: 0.75rem;
        border-radius: 6px;
        border: none;
    }
    
    /* Table */
    .stDataFrame {
        border: 1px solid #ececf1 !important;
        border-radius: 6px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d9d9e3;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #b4b4be;
    }
    
    /* Hide Labels */
    .stTextArea label {
        display: none;
    }
    
    /* Form Submit Button Override */
    .stForm button[kind="primary"] {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.625rem 1.25rem !important;
        font-weight: 500 !important;
    }
    
    .stForm button[kind="primary"]:hover {
        background: #1a7f64 !important;
    }
    
    /* Remove extra padding */
    .element-container {
        margin: 0 !important;
    }
    
    /* Status widget styling */
    .stStatus {
        background: #f7f7f8;
        border: 1px solid #ececf1;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Chat Page (Main Interface)
def chat_page():
    # Initialize vectorstore and chains with user-friendly progress
    try:
        # Check cache status
        pdf_path = os.path.join(config.DIRECTORY, 'data-1.pdf')
        faiss_index_path = "faiss_index"
        index_faiss_file = os.path.join(faiss_index_path, 'index.faiss')
        index_pkl_file = os.path.join(faiss_index_path, 'index.pkl')
        
        cache_exists = os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file)
        
        # Show status message
        if not cache_exists:
            if not os.path.exists(pdf_path):
                st.error("📄 **PDF file not found!**")
                st.warning(f"Please place your SRMIST Academic Regulations PDF as `data-1.pdf` in the `{config.DIRECTORY}` folder.")
                st.info("**Expected location:** `./data/data-1.pdf`")
                st.stop()
            else:
                st.info("🔨 **First-time setup:** Building FAISS index from PDF and saving to local disk...")
                st.warning("⏱️ This will take 2-3 minutes. **After this, all future startups will be instant!**")
                with st.spinner("📚 Processing PDF and creating embeddings..."):
                    vectorstore = init_vectorstore()
                st.success("✅ Index created and saved locally! From now on, restarts will be instant!")
                time.sleep(2)
            st.rerun()
        else:
            # Load from cache - will be instant!
            with st.spinner("⚡ Loading from local cache..."):
                vectorstore = init_vectorstore()
        
        # Initialize chains (also cached) - using custom settings if provided
        qa_chain, checker_chain, answer_type_chain = init_chains(
            vectorstore,
            _api_key=st.session_state.custom_api_key,
            _model=st.session_state.custom_model,
            _temperature=st.session_state.custom_temperature
        )
        print(f"✓ Chains initialized: qa_chain={qa_chain is not None}")
        
    except FileNotFoundError as e:
        st.error("📄 **PDF File Not Found**")
        st.warning(str(e))
        st.info("**What to do:**")
        st.markdown("""
        1. Place your SRMIST Academic Regulations PDF in the `./data/` folder
        2. Rename it to `data-1.pdf`
        3. Refresh this page
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error initializing system**")
        st.code(str(e))
        logger.error(f"Initialization error: {str(e)}")
        st.info("**Troubleshooting:**")
        st.markdown("""
        - Make sure `data-1.pdf` exists in the `./data/` folder
        - Try deleting the `faiss_index` folder and restarting
        - Check the logs folder for more details
        """)
        st.stop()
    
    # Minimal header - Production style
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div class="header-container">
            <div class="header-title">📚 SRM Academic Assistant</div>
            <div class="header-subtitle">Ask once. Get the official answer.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("⚙️", key="settings_btn", help="Settings & Info", use_container_width=True):
            st.session_state.show_info = not st.session_state.get('show_info', False)
            st.rerun()
    
    # Settings Panel (minimal)
    if st.session_state.get('show_info', False):
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            
            st.markdown("**System Status**")
            faiss_exists = os.path.exists(os.path.join("faiss_index", "index.faiss"))
            if faiss_exists:
                st.success("✓ Cache Ready")
            else:
                st.info("Building cache...")
            
            st.markdown("---")
            
            # API Configuration Section
            st.markdown("### 🔑 AI Configuration")
            
            # API Key input
            api_key_input = st.text_input(
                "Groq API Key",
                value=st.session_state.custom_api_key,
                type="password",
                help="Enter your Groq API key. Get one at https://console.groq.com"
            )
            
            # Model selection - ALL available Groq models
            available_models = [
                # Llama 3.3 (Latest, Fastest & Most Capable)
                "llama-3.3-70b-versatile",
                "llama-3.3-70b-specdec",
                
                # Llama 4 (Experimental)
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                
                # Llama 3.1 (Stable)
                "llama-3.1-70b-versatile", 
                "llama-3.1-8b-instant",
                
                # Llama 3.2 (Small & Efficient)
                "llama-3.2-1b-preview",
                "llama-3.2-3b-preview",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview",
                
                # Llama 3 (Legacy)
                "llama3-70b-8192",
                "llama3-8b-8192",
                
                # Mixtral (Long Context)
                "mixtral-8x7b-32768",
                
                # Gemma (Google)
                "gemma-7b-it",
                "gemma2-9b-it",
                
                # Llama Guard (Safety)
                "llama-guard-3-8b"
            ]
            
            # Model descriptions
            model_descriptions = {
                "llama-3.3-70b-versatile": "🚀 RECOMMENDED - Latest, fastest, most capable",
                "llama-3.3-70b-specdec": "⚡ Speculative decoding - Ultra fast",
                "meta-llama/llama-4-maverick-17b-128e-instruct": "🔬 Experimental - 128k context",
                "meta-llama/llama-4-scout-17b-16e-instruct": "🔬 Experimental - 16k context",
                "llama-3.1-70b-versatile": "✅ Stable & Reliable",
                "llama-3.1-8b-instant": "⚡ Ultra Fast - Instant responses",
                "llama-3.2-1b-preview": "💨 Tiny - 1B params",
                "llama-3.2-3b-preview": "💨 Small - 3B params",
                "llama-3.2-11b-vision-preview": "👁️ Vision capable - 11B",
                "llama-3.2-90b-vision-preview": "👁️ Vision capable - 90B (best)",
                "llama3-70b-8192": "📜 Legacy - 8k context",
                "llama3-8b-8192": "📜 Legacy - Fast, 8k context",
                "mixtral-8x7b-32768": "📚 Long context - 32k tokens",
                "gemma-7b-it": "🤖 Google Gemma - 7B",
                "gemma2-9b-it": "🤖 Google Gemma 2 - 9B",
                "llama-guard-3-8b": "🛡️ Safety focused"
            }
            
            # Format model options with descriptions
            model_options = [f"{m} - {model_descriptions.get(m, '')}" for m in available_models]
            current_model_display = f"{st.session_state.custom_model} - {model_descriptions.get(st.session_state.custom_model, '')}"
            
            selected_display = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(current_model_display) if current_model_display in model_options else 0,
                help="Select the AI model to use. llama-3.3-70b-versatile is recommended for best quality."
            )
            
            # Extract actual model name from display string
            model_input = selected_display.split(" - ")[0]
            
            # Temperature slider
            temp_input = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.custom_temperature),
                step=0.1,
                help="Lower = more focused, Higher = more creative"
            )
            
            # Apply button
            if st.button("💾 Apply Changes", use_container_width=True):
                settings_changed = (
                    api_key_input != st.session_state.custom_api_key or
                    model_input != st.session_state.custom_model or
                    temp_input != st.session_state.custom_temperature
                )
                
                if settings_changed:
                    st.session_state.custom_api_key = api_key_input
                    st.session_state.custom_model = model_input
                    st.session_state.custom_temperature = temp_input
                    
                    # Clear cache to reinitialize with new settings
                    st.cache_resource.clear()
                    
                    st.success("✅ Settings updated! Reloading...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("No changes detected")
            
            # Reset to defaults
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.session_state.custom_api_key = config.GROQ_API_KEY
                st.session_state.custom_model = config.GROQ_MODEL
                st.session_state.custom_temperature = config.GROQ_TEMPERATURE
                st.cache_resource.clear()
                st.success("✅ Reset to defaults!")
                time.sleep(1)
                st.rerun()
            
            st.markdown("---")
            
            if st.session_state.metrics:
                st.markdown("**Session Stats**")
                st.metric("Questions", len(st.session_state.metrics))
                avg_time = sum([float(m['response_time'].replace('s','')) for m in st.session_state.metrics]) / len(st.session_state.metrics)
                st.metric("Avg Response", f"{avg_time:.2f}s")
                
                if st.button("📥 Export Data"):
                    os.makedirs('exports', exist_ok=True)
                    df = pd.DataFrame(st.session_state.metrics)
                    filename = 'exports/chat_export.xlsx'
                    df.to_excel(filename, index=False)
                    st.success("Exported!")
            
            st.markdown("---")
            
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.metrics = []
                st.rerun()
            
            if st.button("✕ Close", use_container_width=True):
                st.session_state.show_info = False
                st.rerun()
            
            st.markdown("---")
            st.markdown("**🔍 Debug Mode**")
            debug_mode = st.checkbox("Show retrieved chunks", value=st.session_state.get('debug_mode', False))
            if debug_mode != st.session_state.get('debug_mode', False):
                st.session_state.debug_mode = debug_mode
                st.rerun()
            
            if st.session_state.get('debug_mode') and st.session_state.get('last_retrieved'):
                st.markdown("**Last Retrieved Chunks:**")
                for chunk in st.session_state.last_retrieved[:5]:
                    st.text(f"Page {chunk['page']}: {chunk['preview']}...")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            # Production welcome screen
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-icon">📚</div>
                <div class="welcome-title">SRM Academic Assistant</div>
                <div class="welcome-subtitle">
                    Your official guide to SRMIST Academic Regulations 2021.<br>
                    Ask once. Get the official answer.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Example prompts in grid
            col1, col2 = st.columns(2)
            with col1:
                if st.button("What is the minimum CGPA?", key="ex1", use_container_width=True):
                    st.session_state.example_query = "What is the minimum CGPA required for graduation?"
                if st.button("Attendance requirements?", key="ex2", use_container_width=True):
                    st.session_state.example_query = "What are the attendance requirements?"
            with col2:
                if st.button("Re-evaluation process?", key="ex3", use_container_width=True):
                    st.session_state.example_query = "How does the re-evaluation process work?"
                if st.button("Grading system?", key="ex4", use_container_width=True):
                    st.session_state.example_query = "Explain the grading system and grade points"
        
        # Display chat messages - Clean chatbot style
        for idx, msg in enumerate(st.session_state.chat_history):
            role = msg['role']
            content = msg['message']
            
            # Escape HTML and preserve formatting
            content_safe = html.escape(content).replace('\n', '<br>')
            
            if role == 'user':
                st.markdown(f"""
                <div class="message-row user">
                    <div class="message-bubble user">
                        <div class="avatar user-avatar">👤</div>
                        <div class="message-box user">{content_safe}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row assistant">
                    <div class="message-bubble assistant">
                        <div class="avatar assistant-avatar">🤖</div>
                        <div class="message-box assistant">{content_safe}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    # Chat input - Fixed at bottom (ChatGPT style)
    st.markdown("---")
    
    # Use example query if set
    default_value = st.session_state.get('example_query', '')
    if default_value:
        st.session_state.example_query = None  # Clear it after using
    
    # Input form with Enter key support
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message", 
            placeholder="Ask about SRMIST regulations... (Press Ctrl+Enter to send)",
            key="user_input",
            value=default_value,
            height=100,
            max_chars=2000
        )
        
        col1, col2 = st.columns([5, 1])
        with col2:
            submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)
    
    # JavaScript for Ctrl+Enter submission
    st.markdown("""
    <script>
    const textarea = window.parent.document.querySelector('textarea');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                const button = window.parent.document.querySelector('button[kind="primary"]');
                if (button) button.click();
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)
    
    if submit_button and user_input.strip():
        # Verify chains are initialized
        try:
            if 'qa_chain' not in locals() or qa_chain is None:
                st.error("⚠️ System not ready. Please refresh the page.")
                st.stop()
        except NameError:
            st.error("⚠️ System not initialized. Please refresh the page.")
            st.stop()
        
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response - optimized for speed
        with st.spinner("🔍 Searching regulations and generating answer..."):
            try:
                start_time = time.time()
                
                # Prepare chat history (only last 2 exchanges for speed)
                chat_msgs = []
                recent_history = st.session_state.chat_history[-5:-1] if len(st.session_state.chat_history) > 5 else st.session_state.chat_history[:-1]
                for item in recent_history:
                    if item['role'] == 'user':
                        chat_msgs.append(HumanMessage(content=item['message']))
                    else:
                        chat_msgs.append(AIMessage(content=item['message']))
                
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # Log for debugging
                print(f">>> Processing query: {user_input[:50]}...")
                logger.info(f"Processing query: {user_input[:50]}...")
                
                # Invoke the chain (retrieves from data-1.pdf and generates answer)
                print(f">>> Invoking qa_chain...")
                result = qa_chain({
                    "input": user_input,
                    "chat_history": chat_msgs,
                    "user_profile": [],
                    "leave_balance": [],
                    "current_date": [SystemMessage(content=current_date)]
                })
                
                logger.info(f"Response generated in {time.time() - start_time:.2f}s")
                
                end_time = time.time()
                response_time = round(end_time - start_time, 3)
                    
                # Extract answer and add source information with transparency
                if isinstance(result, dict):
                    answer = result.get('answer') or result.get('result') or str(result)
                    retrieved_docs = result.get('context', [])
                    retrieval_count = len(retrieved_docs) if retrieved_docs else 0
                    
                    # Log what was actually retrieved (PROOF of RAG working)
                    logger.info(f"=== RAG RETRIEVAL PROOF ===")
                    logger.info(f"Query: {user_input}")
                    logger.info(f"Retrieved {retrieval_count} chunks from PDF")
                    
                    # Extract unique page numbers and log them
                    pages = set()
                    retrieved_content_summary = []
                    for idx, doc in enumerate(retrieved_docs):
                        if 'page' in doc.metadata:
                            page_num = doc.metadata['page']
                            pages.add(page_num)
                            # Log first 100 chars of each retrieved chunk (ASCII safe)
                            chunk_preview = doc.page_content[:100].replace('\n', ' ')
                            # Remove special characters for Windows console
                            chunk_preview_safe = chunk_preview.encode('ascii', 'ignore').decode('ascii')
                            logger.info(f"  Chunk {idx+1}: Page {page_num} - '{chunk_preview_safe}...'")
                            retrieved_content_summary.append({
                                'page': page_num,
                                'preview': chunk_preview
                            })
                    
                    source_doc_ids = [
                        f"data-1.pdf:Page{doc.metadata.get('page', '?')}" 
                        for doc in retrieved_docs if 'page' in doc.metadata
                    ]
                    
                    # Don't append source - let the LLM include it in the answer naturally
                    # The prompt instructs it to cite sources inline
                    
                    logger.info(f"=== Retrieved from {len(pages)} unique pages ===")
                    
                    # Store retrieved content for debugging
                    if 'debug_mode' not in st.session_state:
                        st.session_state.debug_mode = False
                    if st.session_state.get('debug_mode'):
                        st.session_state.last_retrieved = retrieved_content_summary
                else:
                    answer = str(result)
                    retrieval_count = 0
                    source_doc_ids = []
                
                # Add assistant response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'message': answer,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Store metrics
                st.session_state.metrics.append({
                    'query': user_input,
                    'output': answer,
                    'response_time': f"{response_time}s",
                    'input_tokens': 'N/A',
                    'output_tokens': len(answer.split()),
                    'api_calls': 1,
                    'retrieval_count': retrieval_count,
                    'source_doc_ids': ', '.join(source_doc_ids)
                })
                
                st.rerun()
                    
            except Exception as e:
                st.error("❌ **Error processing your question**")
                st.exception(e)  # Show full error to user
                logger.error(f"Query error: {str(e)}", exc_info=True)
                
                # Remove the failed user message
                if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                    st.session_state.chat_history.pop()

# Main app
def main():
    init_session_state()
    load_custom_css()
    
    # Go directly to chat interface (no authentication)
    chat_page()

if __name__ == "__main__":
    main()

