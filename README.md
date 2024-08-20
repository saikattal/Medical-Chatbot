# Medical-Chatbot

**This is a Flask medical chatbot application that uses RAG, FAISS db as vector store and local llm Llama 2 GGML running in local CPU.**

**Step 1:**

1. Download the model 'llama-2-7b-chat.ggmlv3.q4_0.bin' and keep it in model folder. Update the model_path in app.py.

2. python store_index.py ------> To create vector store

3. python app.py ------> to run the application