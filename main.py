"""
Please use app.py instead:
    streamlit run app.py
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, redirect, url_for, session
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from helper_functions import get_groq_llm, get_embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from chain import create_qa_chain, create_checker_chain, check_answer_type_chain
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from logger import setup_logger
import config
import json
import os
import logging
from datetime import datetime
import traceback
import pandas as pd
import io
from langchain_core.messages.utils import count_tokens_approximately
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import session

# Print deprecation warning
print("=" * 80)
print("⚠️  DEPRECATION WARNING")
print("=" * 80)
print("This Flask application is deprecated. Please use the Streamlit version:")
print("    streamlit run app.py")
print("=" * 80)
print("\nNOTE: Supabase and authentication features have been removed.")
print("The Streamlit version provides a simpler, authentication-free experience.")
print("=" * 80)

logger = setup_logger()
logger.setLevel(logging.INFO)

# Add a stream handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

load_dotenv()
logger.info("Starting the application")
app = Flask(__name__)

# Initialize global variables
vectorstore = None
conversation_chain = None
chat_history = []
directory = config.DIRECTORY

# Supabase removed - no longer needed
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')  # For session

# Remove groq_llm and use get_groq_llm for LLM

def get_chains(vectorstore):
    logger.info("Creating conversation chain")
    llm = get_groq_llm()
    chain = create_qa_chain(llm, vectorstore)
    checker_chain = create_checker_chain(llm)
    answer_type_chain = check_answer_type_chain(llm)
    logger.info("Conversation chain created successfully")
    return chain, checker_chain, answer_type_chain

# def get_checker_chain():
#     logger.info("Creating checker chain")
#     llm2 = initialize_llm()
#     checker_chain = create_checker_chain(llm2)
#     logger.info("Checker chain created successfully")
#     return checker_chain

# def get_answer_type_checker_chain():
#     logger.info("Creating checker chain")
#     llm3 = initialize_llm()
#     answer_type_chain = check_answer_type_chain(llm3)
#     logger.info("Checker chain created successfully")
#     return answer_type_chain

def probation_checker(current_date, profile_dict):
    try:
        probation_end_date_str = profile_dict.get('probationEndDate')
        if not probation_end_date_str:
            return False

        probation_end_date = datetime.strptime(probation_end_date_str, '%Y-%m-%d').date()
        return probation_end_date >= current_date.date()

    except ValueError as e:
        print(f"Error parsing date: {e}")
        return False
            

def get_text_chunks(directory):
    logger.info(f"Processing data-1.pdf in directory: {directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300, length_function=len)
    chunks = []
    file_path = os.path.join(directory, 'data-1.pdf')
    if os.path.exists(file_path):
        loader = PyPDFLoader(file_path)
        document = loader.load()
        logger.info(f"Loaded document from {file_path}")
        doc_chunks = text_splitter.split_documents(document)
        logger.info(f"Split {file_path} into {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    else:
        logger.warning(f"data-1.pdf not found in {directory}")
    logger.info(f"Processed {len(chunks)} total chunks from data-1.pdf")
    return chunks

def create_vectorstore(docs):
    logger.info(f"Creating vectorstore with {len(docs)} documents")
    logger.info(f"First two documents: {docs[:2]}")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("Vectorstore created successfully")
    return vectorstore

logger.info("Processing documents and initializing vectorstore")
faiss_index_path = "faiss_index"
pdf_path = os.path.join(directory, 'data-1.pdf')
rebuild_index = False
if not os.path.exists(faiss_index_path):
    rebuild_index = True
elif os.path.exists(pdf_path):
    pdf_mtime = os.path.getmtime(pdf_path)
    index_mtime = os.path.getmtime(faiss_index_path)
    if pdf_mtime > index_mtime:
        logger.info("data-1.pdf has been updated since the last index build. Rebuilding FAISS index...")
        rebuild_index = True
if rebuild_index:
    docs = get_text_chunks(directory)
    vectorstore = create_vectorstore(docs)
    vectorstore.save_local(faiss_index_path)
    logger.info("Vectorstore created and saved to disk.")
else:
    logger.info("Loading FAISS vectorstore from disk...")
    vectorstore = FAISS.load_local(faiss_index_path, get_embeddings(), allow_dangerous_deserialization=True)
    logger.info("Vectorstore loaded from disk.")
conversation_chain, checker_chain, check_answer_type_chain = get_chains(vectorstore)
print(conversation_chain)
print(checker_chain)
logger.info("Application initialization complete")

def estimate_input_tokens(query, chat_history, user_profile, leave_balance, current_date):
    messages = []
    if chat_history:
        messages.extend(chat_history)
    if user_profile:
        messages.extend(user_profile)
    if leave_balance:
        messages.extend(leave_balance)
    messages.append(SystemMessage(content=current_date))
    messages.append(HumanMessage(content=query))
    # Log breakdown
    import logging
    logger = logging.getLogger("TokenBreakdown")
    logger.info("--- Token Count Breakdown ---")
    total_tokens = 0
    for i, msg in enumerate(messages):
        content = getattr(msg, 'content', str(msg))
        tokens = count_tokens_approximately([msg])
        logger.info(f"Message {i+1} ({msg.__class__.__name__}): '{content[:60]}...' => {tokens} tokens")
        total_tokens += tokens
    logger.info(f"TOTAL input tokens: {total_tokens}")
    logger.info("----------------------------")
    return total_tokens

@app.route('/generate', methods=['POST'])
def query():
    try:
        global conversation_chain, checker_chain, chat_history, check_answer_type_chain
        # Remove print debug statements
        logger.info("Received /generate request")
        
        # Validate request data
        if not request.is_json:
            logger.error("Request does not contain JSON data")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        logger.info(f"Request data received: {json.dumps(data, indent=2)}")
        
        if conversation_chain is None:
            logger.error("Conversation chain not initialized")
            return jsonify({"error": "System not properly initialized"}), 500
        
        # Validate required fields
        query = data.get('query')
        if not query:
            logger.error("Missing required field: query")
            return jsonify({"error": "Query is required"}), 400
        
        # Process user profile
        user_profile = []
        if 'user_profile' in data:
            try:
                user_profile = [HumanMessage(content=json.dumps(data['user_profile']))]
                logger.info(f"User profile processed: {user_profile}")
            except Exception as e:
                logger.warning(f"Error processing user profile: {str(e)}")
        
        leave_balance = []
        if 'leave_balance' in data:
            try:
                leave_balance = [HumanMessage(content=json.dumps(data['leave_balance']))]
                logger.info(f"User leaves processed: {leave_balance}")
            except Exception as e:
                logger.warning(f"Error processing user profile: {str(e)}")
        
        # Process chat history
        chat_history = []
        if data.get('chat_history'):
            try:
                data["chat_history"] = json.loads(data.get('chat_history'))
                chat_history = []
                for item in data["chat_history"]:
                    if "message" in item and "role" in item:
                        if item["role"] == "USER":
                            chat_history.append(HumanMessage(content=item["message"]))
                        elif item["role"] in ["CHATBOT", "ASSISTANT"]:
                            chat_history.append(AIMessage(content=item["message"]))
                logger.info("Chat history processed successfully")
                logger.info(chat_history)
            except Exception as e:
                logger.warning(f"Error processing chat history: {str(e)}")
        
        # Generate response
        current_date = datetime.now().strftime("%Y-%m-%d")
        probation_flag = " I am still under probation." if probation_checker(current_date, data) else ""
        query += probation_flag
        logger.info(f"Probation Status: {query}")
        try:
            # --- Start timing for response time metric ---
            import time
            start_time = time.time()
            # ---
            result = conversation_chain.invoke({
                "input": query,
                "chat_history": chat_history,
                "user_profile": user_profile,
                "leave_balance": leave_balance,
                "current_date": [SystemMessage(content=current_date)]
            })
            # --- End timing ---
            end_time = time.time()
            response_time = round(end_time - start_time, 3)  # seconds
            logger.info(f"Original RAG response:{result}")
            # Try to extract token usage and output tokens if available
            token_usage = None
            output_tokens = None
            input_tokens = None
            retrieval_count = None
            source_doc_ids = None
            # If result is a dict with these fields, extract them
            if isinstance(result, dict):
                answer = result.get('answer') or result.get('result') or str(result)
                token_usage = result.get('token_usage')
                output_tokens = result.get('output_tokens')
                input_tokens = result.get('input_tokens')
                # Extract retrieval metrics from 'context' key (LCEL chains)
                retrieved_docs = result.get('context')
                if retrieved_docs is not None and isinstance(retrieved_docs, list):
                    retrieval_count = len(retrieved_docs)
                    source_doc_ids = [
                        doc.metadata.get('source', '') + (f":{doc.metadata.get('page', '')}" if 'page' in doc.metadata else '')
                        for doc in retrieved_docs
                    ]
            else:
                answer = str(result)
            # Try to get token usage from LLM if available
            if hasattr(conversation_chain, 'llm') and hasattr(conversation_chain.llm, 'get_last_token_usage'):
                try:
                    token_usage = conversation_chain.llm.get_last_token_usage()
                except Exception:
                    pass
            # Fallback: estimate output tokens
            if not output_tokens and answer:
                output_tokens = len(answer.split())
            if not input_tokens:
                input_tokens = estimate_input_tokens(query, chat_history, user_profile, leave_balance, current_date)
            response = {
                "answer": answer,
                "token_usage": token_usage,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "retrieval_count": retrieval_count,
                "source_doc_ids": source_doc_ids,
                "response_time": response_time  # now in seconds
            }
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/export_excel', methods=['POST'])
def export_excel():
    data = request.get_json()
    exchanges = data.get('exchanges', [])
    if not exchanges:
        return {'message': 'No exchanges to export.'}, 400
    df_new = pd.DataFrame([
        {
            'Query': ex.get('query', ''),
            'Output': ex.get('output', ''),
            'Response Time': ex.get('response_time', ''),
            'Input Tokens': ex.get('input_tokens', ''),
            'Output Tokens': ex.get('output_tokens', ''),
            'API Calls': ex.get('api_calls', ''),
            'Retrieval Count': ex.get('retrieval_count', ''),
            'Source Document IDs': ', '.join(ex.get('source_doc_ids', [])) if isinstance(ex.get('source_doc_ids'), list) else ex.get('source_doc_ids', '')
        } for ex in exchanges
    ])
    # Ensure exports directory exists
    os.makedirs('exports', exist_ok=True)
    filename = 'exports/chat_export.xlsx'
    # If file exists, append (without duplicating previous rows)
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        # Only append rows that are not already present (by Query+Output+Response Time)
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Drop duplicates, keeping the first occurrence
        combined = combined.drop_duplicates(subset=['Query', 'Output', 'Response Time'], keep='first')
        combined.to_excel(filename, index=False)
    else:
        df_new.to_excel(filename, index=False)
    return {'message': f'Exported to {filename}'}, 200

@app.route('/feedback', methods=['POST'])
def feedback():
    """Feedback endpoint - DEPRECATED (Supabase removed)"""
    logger.warning("Feedback endpoint called but Supabase has been removed")
    return jsonify({'status': 'success', 'message': 'Feedback feature disabled'})

@app.route('/rerun', methods=['POST'])
def rerun():
    try:
        # Accept same payload as /generate
        data = request.json
        logger.info(f"Received rerun request: {json.dumps(data, indent=2)}")
        # Reuse the /generate logic
        # (Could refactor to avoid code duplication)
        # For now, just call the query() logic
        # This is a simple way to allow rerun from frontend
        with app.test_request_context('/generate', method='POST', json=data):
            return query()
    except Exception as e:
        logger.error(f"Error in /rerun: {str(e)}")
        return jsonify({'status': 'error', 'details': str(e)}), 500

# In-memory user store for demo
users = {}

@app.route('/')
def landing():
    # If user is logged in, show 'Back to Chatbot' button
    if session.get('user_email'):
        return render_template('landing.html') + "<script>document.getElementById('cta-buttons').style.display='none';document.getElementById('back-btn').style.display='block';</script>"
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and users[email] == password:
            session['user_email'] = email
            return redirect('/chat')
        else:
            return render_template('login.html') + "<script>alert('Invalid credentials');</script>"
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if not email or not password or not confirm or password != confirm:
            return render_template('signup.html') + "<script>alert('Passwords do not match or missing fields');</script>"
        if email in users:
            return render_template('signup.html') + "<script>alert('Account already exists');</script>"
        users[email] = password
        session['user_email'] = email
        return redirect('/chat')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect('/')

@app.route('/chat')
def chat_ui():
    # Only allow access if logged in
    if not session.get('user_email'):
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.errorhandler(400)
def bad_request(e):
    try:
        logger.error(f"Bad request: {str(e)}")
        return jsonify({"error": "Bad request", "details": str(e)}), 400
    except Exception as e:
        logger.error(e)

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Processing documents and initializing vectorstore")
        faiss_index_path = "faiss_index"
        pdf_path = os.path.join(directory, 'data-1.pdf')
        rebuild_index = False
        if not os.path.exists(faiss_index_path):
            rebuild_index = True
        elif os.path.exists(pdf_path):
            pdf_mtime = os.path.getmtime(pdf_path)
            index_mtime = os.path.getmtime(faiss_index_path)
            if pdf_mtime > index_mtime:
                logger.info("data-1.pdf has been updated since the last index build. Rebuilding FAISS index...")
                rebuild_index = True
        if rebuild_index:
            docs = get_text_chunks(directory)
            vectorstore = create_vectorstore(docs)
            vectorstore.save_local(faiss_index_path)
            logger.info("Vectorstore created and saved to disk.")
        else:
            logger.info("Loading FAISS vectorstore from disk...")
            vectorstore = FAISS.load_local(faiss_index_path, get_embeddings(), allow_dangerous_deserialization=True)
            logger.info("Vectorstore loaded from disk.")
        # conversation_chain = get_conversation_chain(vectorstore)
        logger.info("Application initialization complete")
        
        app.run(host='0.0.0.0', port=3000)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}\n{traceback.format_exc()}")
        raise


