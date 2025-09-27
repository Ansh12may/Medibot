# src/helper.py

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# PDF Loader
def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Directory Loader
def load_directory(directory_path):
    loader = DirectoryLoader(directory_path, glob="**/*.pdf")
    return loader.load()

# Text splitter (optional, simple example)
def text_split(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
    return chunks

# HuggingFace embeddings
def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
