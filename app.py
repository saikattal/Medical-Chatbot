from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.vectorstores import FAISS


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
HF_TOKEN=os.environ.get('HF_TOKEN')

embeddings = download_huggingface_embeddings()

vstore = FAISS.load_local("C:\\Users\\LENOVO\\Documents\\LLMOPs Projects\\Medical-Chatbot\\db", embeddings,"medchat-db")



PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

model_path="C:\\Users\\LENOVO\\Documents\\LLMOPs Projects\\Medical-Chatbot\\model\\llama-2-7b-chat.ggmlv3.q4_0.bin"


llm=CTransformers(model=model_path,
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(port=8080,debug= True)
