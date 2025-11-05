import os
import requests
from langchain_groq import ChatGroq
import config
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def get_groq_llm(api_key=None, model=None, temperature=None):
    """
    Returns a ChatGroq instance using config.py values or custom parameters.
    """
    return ChatGroq(
        api_key=api_key if api_key else config.GROQ_API_KEY,
        model=model if model else config.GROQ_MODEL,
        temperature=temperature if temperature is not None else config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS
    )

# TODO: Add Groq embeddings helper if/when available
