import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Modern White & Gold UI Styling ---
st.set_page_config(page_title="PropTech AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .main-title { 
        background: linear-gradient(90deg, #1e3a8a, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 40px; font-weight: 800; text-align: center; margin-bottom: 5px;
    }
    .sub-title { color: #64748b; text-align: center; font-size: 16px; margin-bottom: 30px; }
    .stChatMessage { 
        background-color: #f8fafc !important; 
        border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px; margin-bottom: 10px;
        color: #1e293b !important; 
    }
    [data-testid="stChatMessageUser"] { border-left: 5px solid #d4af37 !important; }
    .stChatInputContainer { border-radius: 10px !important; border: 1px solid #cbd5e1 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 PropTech AI Consultant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Your Smart Partner in Real Estate & Investment</div>", unsafe_allow_html=True)

# --- Auto-Ingestion Logic ---
@st.cache_resource
def load_db():
    DB_PATH = "./chroma_db_prop"
    if not os.path.exists(DB_PATH):
        with st.spinner("⏳ First run: Building Property Knowledge Base..."):
            from ingestion import create_prop_db
            create_prop_db()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={'k': 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --- Prompt ---
template = """You are an expert Real Estate Consultant. Use the provided context 
to answer the user's question about property, investment, or taxes. 
Context: {context}
Question: {question}
Expert Advice:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about investment, taxes, or property guides..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing property documents..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            
            # References
            docs = retriever.invoke(query)
            with st.expander("🔍 Legal & Market Sources"):
                sources = {doc.metadata.get('source') for doc in docs}
                for s in sources: st.write(f"📍 {s}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

            