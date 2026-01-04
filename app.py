import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Configuration ---
DATA_PATH = "prop_data/"
DB_PATH = "chroma_db_prop"

# --- Internal Automatic Ingestion ---
def setup_prop_knowledge():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return False
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    if not raw_docs:
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    return True

# --- UI Styling (Luxury White & Gold) ---
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
    .stChatMessage p, .stChatMessage div { color: #1e293b !important; }
    [data-testid="stChatMessageUser"] { border-left: 5px solid #d4af37 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 PropTech AI Consultant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Luxury Real Estate Intelligence & Universal Property Advisor</div>", unsafe_allow_html=True)

# --- Database Loading ---
@st.cache_resource
def get_prop_retriever():
    if not os.path.exists(DB_PATH):
        with st.spinner("⏳ First run: Building Property Knowledge Base..."):
            success = setup_prop_knowledge()
            if not success: return None
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={'k': 4})

retriever = get_prop_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- UNIVERSAL HYBRID PROMPT ---
template = """You are an elite Real Estate and Investment Consultant.

1. PROPERTY KNOWLEDGE: Use the provided Context to answer specific questions about property taxes, investment guides, or market analytics found in the documents.
2. WORLD KNOWLEDGE: If the answer is not in the documents, or if the user asks about world news, current leaders, geography, or general topics, use your internal intelligence to provide a professional response.
3. BE UNIVERSAL: Do not limit your knowledge to one city. Provide expert advice for any global query.
4. IDENTITY: You are a RAG-based AI powered by GPT-4o-mini.

Context: {context}
Question: {question}

Expert Advice:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No property documents found."
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs)

if retriever:
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
else:
    rag_chain = ({"context": lambda x: "None", "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about investment, world news, or property guides..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing property & world data..."):
            try:
                response = rag_chain.invoke(query)
                st.markdown(response)
                
                # References
                if retriever:
                    docs = retriever.invoke(query)
                    with st.expander("🔍 Legal & Market Sources"):
                        sources = {doc.metadata.get('source') for doc in docs}
                        for s in sources: st.write(f"📍 {s}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                