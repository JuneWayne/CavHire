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
PINECONE_ENV = os.getenv("PINECONE_ENV")  

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in your environment.")
    st.stop()
if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Missing Pinecone API key or environment in your environment.")
    st.stop()

openai.api_key = OPENAI_API_KEY

INDEX_NAME = "csv-data-index"

pinecone.Index = pinecone.data.index.Index

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Index {INDEX_NAME} not found. Please run ingestion.py first.")
    st.stop()

openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = PineconeStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=lambda x: openai_embedding.embed_query(x)
)

# -----------------------------
# Create Conversational Chat Model using OpenAI
# -----------------------------
chat_model = ChatOpenAI(temperature=0.2, model_name="gpt-4o", verbose=True)

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
You are Layla, an expert assistant helping students find job opportunities from the UVA Student Jobs database.

Your goal is to provide **clear, accurate, and well-structured answers** using ONLY the information provided in the context below. If the context does not contain the necessary information to answer the question, be cautious and reply with 'sorry, I'm not sure'

When answering:

1. If the question is about a **specific job**, extract and organize the following details (if available):
   - **Job Title**
   - **Location**
   - **Salary**
   - **Job Description**
   - **Key Responsibilities**
   - **Requirements and Qualifications**
   - **Application Instructions**

2. If the question is **general** (e.g., "What jobs are available?"), return a concise list of **multiple job postings**, each structured as a bullet point with the same fields above (as available).

3. Be factual, concise, and do NOT speculate. Do not add information that is not explicitly mentioned in the context.

4. Use bullet points or headings to keep your response organized and easy to read.

---

**Context:**
{context}

**User Question:**
{question}

---

**Answer:**
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
    st.title("UVA Student Jobs Copilot 😽")
    #st.image("../html_files/html_data/uva.png", width=45)
    st.subheader("Ask Layla about part-time opportunities currently available!")
    
    if "chat_history" not in st.session_state:

        st.session_state.chat_history = [
             {"role": "assistant", "content": "Hello! What opportunities are you looking for today?"}
        ]


    conv_chain = create_conversational_chain(llm=chat_model, retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}))
    
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_message = st.chat_input("Type your message here...")
    if user_message:
    
        st.chat_message("user").write(user_message)
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        with st.spinner("Layla is thinking..."):
            response = conv_chain({
                "question": user_message,
                "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]
            })
            answer = response.get("answer", "No answer found.")
        
      
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        

if __name__ == "__main__":
    main()
