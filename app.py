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



import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama  # Import Llama for local inference

# ✅ Set Page Config FIRST to avoid Streamlit errors
st.set_page_config(page_title="Legal RAG System", layout="wide")

# 🔹 Load API Keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

INDEX_NAME = "lawdata-index"

# 🔹 Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Check if Index Exists
if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

index = pc.Index(INDEX_NAME)
st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# 🔹 Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Load Llama Model
llm = Llama(model_path="path/to/llama-model.bin")  # Update path to your model

# 🔹 Streamlit UI
st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")

# 🔹 User Input
query = st.text_input("🔍 Enter your legal question:")

if query:
    with st.spinner("🔎 Searching for relevant legal documents..."):
        try:
            # Generate query embedding
            query_embedding = embedding_model.encode(query).tolist()

            # Retrieve top 10 relevant documents
            search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

            if search_results.get("matches"):
                # Rank results based on similarity score
                ranked_chunks = sorted(search_results["matches"], key=lambda x: x["score"], reverse=True)
                
                # Extract top relevant texts
                context_chunks = [match["metadata"]["text"] for match in ranked_chunks]
                context_text = "\n\n".join(context_chunks)[:2000]  # Limit context length

                # Display retrieved documents
                with st.expander("📄 Retrieved Documents (Top 10 Chunks)"):
                    for i, chunk in enumerate(context_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        st.info(chunk)

                # 🔹 Prepare AI Prompt
                prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.
                
                Context:
                {context_text}
                
                Question: {query}
                
                Answer:"""
                
                # 🔹 Generate Response Using Llama
                response = llm(prompt)
                answer = response["choices"][0]["text"].strip()
                
                st.success("💡 AI Response:")
                st.write(answer)
            else:
                st.warning("⚠️ No relevant legal documents found. Try rephrasing your query.")
        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# 🔹 Footer
st.markdown("---")
st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama** for Legal AI Assistance.")

