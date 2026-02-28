# Medical Chatbot

Medibot is an AI-powered medical chatbot built using the Retrieval-Augmented Generation (RAG) architecture.
It provides context-aware medical responses by retrieving relevant information from a medical dataset and enhancing large language model (LLM) outputs with domain-specific knowledge.

Unlike traditional chatbots that rely only on pre-trained knowledge, Medibot retrieves verified medical content before generating responses, improving accuracy and reliability.

 Architecture

Medibot follows the standard RAG pipeline:

Document Ingestion

Medical dataset (clinical notes / PDFs / structured files)

Text chunking

Embedding generation

Vector Database

Stores document embeddings

Enables semantic similarity search

Retriever

Fetches top-k relevant chunks based on query

LLM Generator

Uses OpenAI API

Generates response using retrieved context



User Interface
Flask 

Tech Stack
Python
OpenAI API
Pinecone
LangChain 
Flask 
Environment Variables (.env)
