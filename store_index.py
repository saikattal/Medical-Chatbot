from src.helper import load_pdf, split_text, download_huggingface_embeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()


extracted_data = load_pdf("data/")
text_chunks = split_text(extracted_data)
embeddings = download_huggingface_embeddings()

vstore= FAISS.from_documents(text_chunks,embeddings)

vstore.save_local("C:\\Users\\LENOVO\\Documents\\LLMOPs Projects\\Medical-Chatbot\\db","medchat-db")



