# import streamlit as st
# import requests
# import pinecone
# from sentence_transformers import SentenceTransformer

# # ✅ Set Page Config FIRST to avoid Streamlit errors
# st.set_page_config(page_title="Legal RAG System", layout="wide")

# # 🔹 Load API Keys from Streamlit Secrets
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# PINECONE_ENV = st.secrets["PINECONE_ENV"]
# TOGETHER_AI_API_KEY = st.secrets.get("TOGETHER_AI_API_KEY")  # Optional API Key

# INDEX_NAME = "lawdata-index"

# # 🔹 Initialize Pinecone Client
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# # 🔹 Check if Index Exists
# if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
#     st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
#     st.stop()

# index = pc.Index(INDEX_NAME)
# st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# # 🔹 Load Embedding Model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # 🔹 Streamlit UI
# st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")

# # 🔹 User Input
# query = st.text_input("🔍 Enter your legal question:")

# if query:
#     with st.spinner("🔎 Searching for relevant legal documents..."):
#         try:
#             # Generate query embedding
#             query_embedding = embedding_model.encode(query).tolist()

#             # Retrieve top 5 relevant documents
#             search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

#             if search_results.get("matches"):
#                 # Extract relevant text
#                 context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]
#                 context_text = "\n\n".join(context_chunks)

#                 # Display retrieved documents
#                 with st.expander("📄 Retrieved Documents (Top 10 Chunks)"):
#                     for i, chunk in enumerate(context_chunks):
#                         st.write(f"**Chunk {i+1}:**")
#                         st.info(chunk)

#                 # 🔹 Prepare AI Prompt
#                 prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.

#                 Context:
#                 {context_text}

#                 Question: {query}

#                 Answer:"""

#                 # 🔹 Call Together AI API
#                 api_url = "https://api.together.xyz/v1/chat/completions"
#                 headers = {"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"}
#                 payload = {
#                     "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                     "messages": [
#                         {"role": "system", "content": "You are an expert in legal matters."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "temperature": 0.2
#                 }

#                 response = requests.post(api_url, headers=headers, json=payload)

#                 if response.status_code == 200:
#                     answer = response.json()["choices"][0]["message"]["content"]
#                     st.success("💡 AI Response:")
#                     st.write(answer)
#                 else:
#                     st.error(f"⚠️ API Error: {response.text}")

#             else:
#                 st.warning("⚠️ No relevant legal documents found. Try rephrasing your query.")

#         except Exception as e:
#             st.error(f"⚠️ Error: {str(e)}")

# # 🔹 Footer
# st.markdown("---")
# st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")

import os
import time
import re
import fitz  # PyMuPDF
import nltk
import cv2
import numpy as np
import pytesseract
import requests
import streamlit as st
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

# ✅ Set Page Config FIRST to avoid Streamlit errors
st.set_page_config(page_title="Legal RAG System", layout="wide")

# 🔹 Load API Keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TOGETHER_AI_API_KEY = st.secrets.get("TOGETHER_AI_API_KEY")
INDEX_NAME = "lawdata-index"

# 🔹 Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Check if Index Exists
if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

index = pc.Index(INDEX_NAME)
st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# 🔹 Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
retrieval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(index_name=INDEX_NAME, embedding=retrieval_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

# 🔹 Streamlit UI
st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")
query = st.text_input("🔍 Enter your legal question:")

if query:
    with st.spinner("🔎 Searching for relevant legal documents..."):
        try:
            query_embedding = embedding_model.encode(query).tolist()
            search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

            if search_results.get("matches"):
                context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]
                context_text = "\n\n".join(context_chunks)

                with st.expander("📄 Retrieved Documents (Top 10 Chunks)"):
                    for i, chunk in enumerate(context_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        st.info(chunk)

                prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.
                
                Context:
                {context_text}
                
                Question: {query}
                
                Answer:"""

                api_url = "https://api.together.xyz/v1/chat/completions"
                headers = {"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert in legal matters."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }

                response = requests.post(api_url, headers=headers, json=payload)

                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.success("💡 AI Response:")
                    st.write(answer)
                else:
                    st.error(f"⚠️ API Error: {response.text}")
            else:
                st.warning("⚠️ No relevant legal documents found. Try rephrasing your query.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

st.markdown("---")
st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")


