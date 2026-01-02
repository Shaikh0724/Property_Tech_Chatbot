import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load Environment Variables
load_dotenv()

# Configuration
DATA_PATH = "property_data/"
DB_PATH = "chroma_db_prop"

def create_prop_db():
    # 1. Check if folder exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created '{DATA_PATH}' folder. Please add property PDFs.")
        return

    # 2. Load Documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    
    if len(raw_docs) == 0:
        print("⚠️ No PDFs found in folder!")
        return

    # 3. Split Text (Larger chunks for legal/tax context)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    documents = text_splitter.split_documents(raw_docs)
    
    # 4. Create Embeddings & Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=DB_PATH
    )
    print(f"🏠 Property Database created with {len(documents)} chunks.")

if __name__ == "__main__":
    create_prop_db()