# src/store_index.py

import os
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings,load_directory
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
# Get Pinecone API key from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index if it doesn't exist
index_name = "medicalbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # adjust based on your embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Example: Load documents and split
docs = load_directory("/research/Data/pdfs")  # adjust path
chunks = text_split(docs)

# Now you can upsert into Pinecone index
index = pc.index(index_name)
for i, chunk in enumerate(chunks):
    index.upsert(
        vectors=[{
            "id": f"chunk-{i}",
            "values": embeddings.embed_text(chunk)
        }]
    )

print(f"Indexed {len(chunks)} chunks into Pinecone '{index_name}'")
