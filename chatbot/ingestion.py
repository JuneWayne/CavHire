import os
import warnings
import pandas as pd
from dotenv import load_dotenv
import pinecone

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore

warnings.filterwarnings("ignore")
load_dotenv('../.env')  # adjust path as needed

# Configuration
CSV_FILE = "../datacollection/scraped_jobs.csv"  # Your CSV file path
INDEX_NAME = "csv-data-index"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west-2"

if not PINECONE_API_KEY or not PINECONE_ENV:
    print("Missing Pinecone API key or environment in your environment.")
    exit(1)

# --- Initialize Pinecone using the new API ---
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI text-embedding-ada-002 outputs 1536-d vectors
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',           # Adjust if using a different cloud provider
            region=PINECONE_ENV    # e.g., "us-west-2"
        )
    )

# --- Workaround for LangChain compatibility ---
# LangChain expects the index object to be of type pinecone.Index;
# in the new Pinecone client, set the attribute accordingly.
pinecone.Index = pinecone.data.index.Index

# --- Helper: Load CSV as Documents ---
def load_documents_from_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    docs = []
    # Use the "text" column if it exists; otherwise, join all columns.
    if "text" in df.columns:
        for _, row in df.iterrows():
            content = str(row["text"])
            metadata = row.to_dict()
            docs.append(Document(page_content=content, metadata=metadata))
    else:
        for _, row in df.iterrows():
            # Concatenate all columns into one string
            content = " ".join([str(val) for val in row.values])
            metadata = row.to_dict()
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

# --- Main ingestion function ---
def create_vector_store():
    print(f"Loading documents from {CSV_FILE} ...")
    docs = load_documents_from_csv(CSV_FILE)
    print(f"Loaded {len(docs)} documents from CSV.")
    
    # Split each document into smaller chunks if needed
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} document chunks.")
    
    # Create embeddings using OpenAI's embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("Storing embeddings in Pinecone...")
    vectorstore = PineconeStore.from_documents(split_docs, embeddings, index_name=INDEX_NAME)
    print("Vector store created successfully!")
    return vectorstore

def main():
    vs = create_vector_store()
    print("Ingestion complete. Your vector store is ready in Pinecone.")

if __name__ == "__main__":
    main()
