import os
import warnings
import streamlit as st
from dotenv import load_dotenv
import pinecone
import openai

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
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

INDEX_NAME = "uva-jobs"

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
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o", verbose=True)

# -----------------------------
# CrewAI Setup: Define Agents, Manager, and Tasks
# -----------------------------
from crewai import Agent, Task, Crew

# Define individual agents for specific tasks.
summarizer_agent = Agent(
    name="Job Summarizer",
    role="Summarizes detailed job information",
    goal="Extract and present complete job details (title, location, requirements, salary, etc.) from the context.",
    backstory="A job description expert with deep understanding of UVA job postings.",
    allow_delegation=False,
    verbose=True
)

qa_agent = Agent(
    name="Concise QA",
    role="Provides brief and accurate answers",
    goal="Answer user questions about jobs with short, direct responses.",
    backstory="A quick responder trained to extract key information from job posts.",
    allow_delegation=False,
    verbose=True
)

recommender_agent = Agent(
    name="Job Recommender",
    role="Recommends jobs based on a user's profile",
    goal="Match the user's qualities and interests with the most suitable UVA job opportunities.",
    backstory="A career coach experienced in tailoring recommendations to student profiles.",
    allow_delegation=False,
    verbose=True
)

# Manager agent: This agent uses an LLMChain (with a custom prompt) to decide
# which agent should handle the request.
manager_prompt = PromptTemplate(
    input_variables=["user_question", "user_profile", "context"],
    template="""
You are the manager AI responsible for delegating a job-related request. Analyze the following:

User Question: {user_question}

User Profile: {user_profile}

Job Context:
{context}

Based on the full context, decide which of the following agents should handle this request:
1. Job Summarizer – for detailed, full job descriptions.
2. Concise QA – for short, direct answers.
3. Job Recommender – for personalized job recommendations based on the user's profile.

Do not rely solely on keywords. Instead, evaluate the overall intent and detail in the query. Return only the chosen agent name exactly as one of:
"Job Summarizer", "Concise QA", or "Job Recommender".
Your decision:
"""
)
manager_chain = LLMChain(llm=chat_model, prompt=manager_prompt)

def run_crewai_with_manager(user_question, user_profile="Data science student looking for flexible tech roles"):
    # Retrieve relevant documents from the vector store
    docs = vectorstore.similarity_search(user_question, k=10)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use the manager chain to decide which agent should handle the request.
    selected_agent_name = manager_chain.run({
        "user_question": user_question,
        "user_profile": user_profile,
        "context": context
    }).strip()

    # Based on the manager's decision, create an appropriate Task.
    if selected_agent_name == "Job Summarizer":
        selected_agent = summarizer_agent
        task_description = f"Summarize the following job context in detail to answer the query: {user_question}"
        expected_output = "A detailed summary of the job including title, location, salary, requirements, etc."
    elif selected_agent_name == "Job Recommender":
        selected_agent = recommender_agent
        task_description = f"Based on the user's profile: '{user_profile}', recommend suitable UVA job opportunities from the context."
        expected_output = "A list of job recommendations with explanations."
    elif selected_agent_name == "Concise QA":
        selected_agent = qa_agent
        task_description = f"Provide a brief, direct answer to the following question using the context: {user_question}"
        expected_output = "A concise 1-2 sentence answer."
    else:
        # Fallback to QA agent if the manager's decision is unclear.
        selected_agent = qa_agent
        task_description = f"Provide a brief, direct answer to the following question using the context: {user_question}"
        expected_output = "A concise 1-2 sentence answer."

    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=selected_agent,
        input=context
    )

    # Create a Crew with just the selected agent and task.
    crew = Crew(agents=[selected_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result

# -----------------------------
# Main Chatbot Interface using Streamlit and CrewAI Manager
# -----------------------------
def main():
    st.title("UVA Student Jobs Chatbot 🐈")
    st.subheader("Chat with Layla about job opportunities on the UVA Student Jobs website!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! What opportunities are you looking for today?"}
        ]

    # Display chat history.
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_message = st.chat_input("Type your message here...")
    if user_message:
        st.chat_message("user").write(user_message)
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Optionally, let the user provide their profile.
        # Here it is hard-coded; you could enhance this by having a profile input field.
        user_profile = "Data science student looking for flexible tech roles"

        with st.spinner("Layla is thinking..."):
            answer = run_crewai_with_manager(user_message, user_profile)
            # Extract and render markdown from CrewAI result
            if isinstance(answer, dict):
                display_text = answer.get("raw", str(answer))
            elif isinstance(answer, str):
                display_text = answer
            else:
                display_text = str(answer)

            st.chat_message("assistant").markdown(display_text)
            st.session_state.chat_history.append({"role": "assistant", "content": display_text})

        
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
