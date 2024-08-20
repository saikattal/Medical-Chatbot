from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

def split_text(extracted_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=20)
    
    text_chunks = text_splitter.split_documents(extracted_documents)
    return text_chunks

def download_huggingface_embeddings():
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings



