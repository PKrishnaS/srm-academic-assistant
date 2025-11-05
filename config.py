import streamlit as st
import os

# Common Configuration
DIRECTORY = './data'
CHUNK_SIZE = 1500
TOP_K = 8  # Reduced for faster retrieval (was 20)

# Groq AI Configuration - supports local and cloud deployment
try:
    # Try Streamlit Cloud secrets first
    GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]
    GROQ_MODEL = st.secrets.get("groq", {}).get("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE = float(st.secrets.get("groq", {}).get("GROQ_TEMPERATURE", 0.5))
    GROQ_MAX_TOKENS = int(st.secrets.get("groq", {}).get("GROQ_MAX_TOKENS", 2048))
except:
    # Fallback to hardcoded (local development only)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_TEMPERATURE = 0.5
    GROQ_MAX_TOKENS = 2048

# HuggingFace Embeddings Configuration
HF_EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
HF_EMBEDDINGS_DEVICE = "cpu"