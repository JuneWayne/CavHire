import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# Load environment variables
load_dotenv('../.env')

# Pull from .env or fall back
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME") or "csv-jobdata-index"

print(f"🔍 Using Pinecone index: {INDEX_NAME}")

# 1. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print(f"🛠️ Creating new index '{INDEX_NAME}' with dimension 1536...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 3. Get the actual index object
index = pc.Index(INDEX_NAME)  # <-- This MUST return an Index object from pinecone-py v3

# 4. Set up embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# 5. Load and embed CSV data
csv_files = [
    "../datacollection/scraped_jobs.csv",
]

for csv_file in csv_files:
    print(f"\n📄 Loading {csv_file}...")
    loader = CSVLoader(file_path=csv_file)
    docs = loader.load()

    # Optional chunking
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks ready for upload.")

    # 6. Store in Pinecone using LangChain’s wrapper
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index=index,           # <-- must be pinecone-py v3 Index object
        namespace="default"
    )

    print(f"✅ Uploaded chunks to index: {INDEX_NAME}")
