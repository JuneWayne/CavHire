import os
import warnings
import streamlit as st
from dotenv import load_dotenv
import pinecone
import openai

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv('../.env')

# -----------------------------
# Configuration & Initialization
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west-2"

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in your environment.")
    st.stop()
if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Missing Pinecone API key or environment in your environment.")
    st.stop()

openai.api_key = OPENAI_API_KEY

INDEX_NAME = "uva-jobs"

# --- Workaround for LangChain compatibility ---
# In the new Pinecone client, the index is of type pinecone.data.index.Index,
# but LangChain expects pinecone.Index. We patch it here:
pinecone.Index = pinecone.data.index.Index

# Initialize Pinecone client using the new API
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Index {INDEX_NAME} not found. Please run ingestion.py first.")
    st.stop()

# Create an instance of OpenAIEmbeddings for retrieval embeddings.
openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load vector store from Pinecone using OpenAI embeddings.
vectorstore = PineconeStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=lambda x: openai_embedding.embed_query(x)
)

# -----------------------------
# Create Conversational Chat Model using OpenAI
# -----------------------------
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", verbose=True)

# -----------------------------
# Conversational Chain Components
# -----------------------------
def create_question_generator(llm):
    question_gen_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question.\n\n"
            "Conversation:\n{chat_history}\n\n"
            "Follow-up question: {question}\n\n"
            "Standalone question:"
        ),
    )
    return LLMChain(llm=llm, prompt=question_gen_prompt)

def create_summary_chain(llm):
    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful job site web-crawler who has scraped open job opportunities from the UVA Student Jobs website.
Your task:
1. Answer the question based on the specific query that the user provided
2. if the user asked a general query, i.e. tell me about jobs available, answer comprehensively and provide all details 

Context:
{context}

User Question:
{question}

Answer:
"""
    )
    return StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=summary_prompt),
        document_variable_name="context"
    )

def create_conversational_chain(llm, retriever):
    question_generator_chain = create_question_generator(llm)
    summary_chain = create_summary_chain(llm)
    return ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=summary_chain
    )

# -----------------------------
# Main Chatbot Interface using Chat UI
# -----------------------------
def main():
    st.title("UVA Student Jobs Chatbot 🐈")
    st.subheader("Chat with Layla about job opportunities on the UVA Student Jobs website!")
    
    # Initialize chat history if not present.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Create conversational chain.
    conv_chain = create_conversational_chain(llm=chat_model, retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}))
    
    # Display previous messages using Streamlit's chat interface.
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Use st.chat_input for new messages.
    user_message = st.chat_input("Type your message here...")
    if user_message:
        # Display the user's message.
        st.chat_message("user").write(user_message)
        # Append the user message to the chat history.
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Run the conversational chain.
        with st.spinner("Layla is thinking..."):
            response = conv_chain({
                "question": user_message,
                "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]
            })
            answer = response.get("answer", "No answer found.")
        
        # Display and store the assistant's response.
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # (Optional) Clear input if desired by re-rendering; st.chat_input clears automatically.

if __name__ == "__main__":
    main()
