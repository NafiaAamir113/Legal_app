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

#             # Retrieve top 10 relevant documents
#             search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

#             if search_results.get("matches"):
#                 # Extract relevant text and document IDs
#                 context_chunks = []
#                 for match in search_results["matches"]:
#                     doc_id = match["id"]  # Get document ID
#                     metadata = match.get("metadata", {})
#                     text = metadata.get("text", "No text available")  # Get text safely
#                     pdf_name = metadata.get("pdf_name", "Unknown PDF")  # Get PDF name safely
                    
#                     context_chunks.append(f"📜 **{pdf_name} (ID: {doc_id})**\n{text}")

#                 context_text = "\n\n".join(context_chunks)

#                 # Display retrieved documents
#                 with st.expander("📄 Retrieved Documents (Top 10 Chunks)"):
#                     for chunk in context_chunks:
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



# import streamlit as st
# import requests
# import pinecone
# from sentence_transformers import SentenceTransformer, CrossEncoder

# st.set_page_config(page_title="Legal RAG System", layout="wide")

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# PINECONE_ENV = st.secrets["PINECONE_ENV"]
# TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# INDEX_NAME = "lawdata-2-index"
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
#     st.error(f"❌ Index '{INDEX_NAME}' not found.")
#     st.stop()

# index = pc.Index(INDEX_NAME)
# embedding_model = SentenceTransformer("BAAI/bge-large-en")
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")

# query = st.text_input("🔍 Enter your legal question:")

# if query:
#     with st.spinner("🔎 Searching..."):
#         query_embedding = embedding_model.encode(query).tolist()
#         search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

#         if search_results.get("matches"):
#             context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]
#             rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
#             ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

#             context_text = "\n\n".join([r[0] for r in ranked_results[:5]])

#             prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.

#             Context:
#             {context_text}

#             Question: {query}

#             Answer:"""

#             response = requests.post(
#                 "https://api.together.xyz/v1/chat/completions",
#                 headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
#                 json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                       "messages": [{"role": "system", "content": "You are an expert in legal matters."},
#                                    {"role": "user", "content": prompt}], "temperature": 0.2}
#             )

#             answer = response.json()["choices"][0]["message"]["content"]
#             st.success("💡 AI Response:")
#             st.write(answer)

# st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")


# # # 🔹 Footer
# # st.markdown("---")
# # st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")







import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit Config
st.set_page_config(page_title="Legal RAG System", layout="wide")

# API Keys
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Initialize Pinecone
INDEX_NAME = "lawdata-2-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found.")
    st.stop()

index = pc.Index(INDEX_NAME)

# Load Models
embedding_model = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# UI Title
st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")

# User Query Input
query = st.text_input("🔍 Enter your legal question:")

# Generate Answer Button
if st.button("🚀 Generate Answer") and query:
    with st.spinner("🔎 Searching..."):
        try:
            # Embedding Search
            query_embedding = embedding_model.encode(query).tolist()
            search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

            if search_results.get("matches"):
                # Extract and Rerank Results
                context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]
                rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
                ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

                # Select dynamic number of chunks (min available or 5)
                num_chunks = min(len(ranked_results), 5)
                context_text = "\n\n".join([r[0] for r in ranked_results[:num_chunks]])

                # Construct LLM prompt
                prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

                Context:
                {context_text}

                Question: {query}

                Answer:"""

                # LLM API Call
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                          "messages": [{"role": "system", "content": "You are an expert in legal matters."},
                                       {"role": "user", "content": prompt}], "temperature": 0.2}
                )

                # Display Response
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.success("💡 AI Response:")
                    st.write(answer)
                else:
                    st.error("⚠️ Failed to generate a response. Please try again.")

            else:
                st.warning("⚠️ No relevant legal documents found. Try refining your question.")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("🚀 Built with **Streamlit**.")


