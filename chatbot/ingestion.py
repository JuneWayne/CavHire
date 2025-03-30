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
load_dotenv('../.env')  

CSV_FILE = "../datacollection/uvajobsdata.csv"  
INDEX_NAME = "csv-data-index"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") 

if not PINECONE_API_KEY or not PINECONE_ENV:
    print("Missing Pinecone API key or environment in your environment.")
    exit(1)

# --- Initialize Pinecone using the new API ---
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',          
            region=PINECONE_ENV   
        )
    )

pinecone.Index = pinecone.data.index.Index

def load_documents_from_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    docs = []
    if "text" in df.columns:
        for _, row in df.iterrows():
            content = str(row["text"])
            metadata = row.to_dict()
            docs.append(Document(page_content=content, metadata=metadata))
    else:
        for _, row in df.iterrows():
        
            content = " ".join([str(val) for val in row.values])
            metadata = row.to_dict()
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def create_vector_store():
    print(f"Loading documents from {CSV_FILE} ...")
    docs = load_documents_from_csv(CSV_FILE)
    print(f"Loaded {len(docs)} documents from CSV.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} document chunks.")
    
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
