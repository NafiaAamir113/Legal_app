# import streamlit as st
# import requests
# import pinecone
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # Streamlit page setup
# st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# # Load secrets
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# # Pinecone setup
# INDEX_NAME = "lawdata-index"
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# # Check if index exists
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if INDEX_NAME not in existing_indexes:
#     st.error(f"Index '{INDEX_NAME}' not found.")
#     st.stop()

# # Initialize Pinecone index
# index = pc.Index(INDEX_NAME)

# # Load embedding models
# embedding_model = SentenceTransformer("BAAI/bge-large-en")
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # Page Title
# st.title("⚖️ LEGAL ASSISTANT")

# # Short App Description
# st.markdown("This AI-powered legal assistant retrieves relevant legal documents and provides accurate responses to your legal queries.")

# # Input field
# query = st.text_input("Enter your legal question:")

# # Generate Answer Button
# if st.button("Generate Answer"):
#     if not query:
#         st.warning("Please enter a legal question before generating an answer.")
#         st.stop()

#     # Check for incomplete query
#     if len(query.split()) < 4:  # Simple heuristic for incomplete queries
#         st.warning("Your query seems incomplete. Please provide more details.")
#         st.stop()

#     with st.spinner("Searching..."):
#         query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

#         # Query Pinecone with error handling
#         try:
#             search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
#         except Exception as e:
#             st.error(f"Pinecone query failed: {e}")
#             st.stop()

#         if not search_results or "matches" not in search_results or not search_results["matches"]:
#             st.warning("No relevant results found. Try rephrasing your query.")
        #     st.stop()

        # # Extract text chunks from results
        # context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

        # # Rerank results
        # rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
        # ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

        # # Select dynamic number of chunks (min available or 5)
        # num_chunks = min(len(ranked_results), 5)
        # context_text = "\n\n".join([r[0] for r in ranked_results[:num_chunks]])

        # # Construct LLM prompt
        # prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

        # Context:
        # {context_text}

        # Question: {query}

        # Answer:"""

        # # Query Together AI
        # response = requests.post(
        #     "https://api.together.xyz/v1/chat/completions",
        #     headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
#             json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                   "messages": [{"role": "system", "content": "You are an expert in legal matters."},
#                                {"role": "user", "content": prompt}], "temperature": 0.2}
#         )

#         answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No valid response from AI.")
#         st.success("AI Response:")
#         st.write(answer)

# # Footer with emoji
# st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)







# import streamlit as st
# import requests
# import pinecone
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from sentence_transformers.util import cos_sim

# # ------------------- Streamlit Page Config -------------------
# st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# # ------------------- Load Secrets -------------------
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]
# HF_TOKEN = st.secrets["HF_TOKEN"]  # Hugging Face Token

# # ------------------- Pinecone Setup -------------------
# INDEX_NAME = "lawdata-index"
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if INDEX_NAME not in existing_indexes:
#     st.error(f"Index '{INDEX_NAME}' not found in Pinecone.")
#     st.stop()

# index = pc.Index(INDEX_NAME)

# # ------------------- Load Embedding Models -------------------
# def load_embedding_model():
#     try:
#         return SentenceTransformer("BAAI/bge-large-en", use_auth_token=HF_TOKEN)
#     except Exception as e:
#         st.warning("⚠️ Could not load 'BAAI/bge-large-en'. Trying fallback model...")
#         try:
#             return SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HF_TOKEN)
#         except Exception as e2:
#             st.error("❌ All embedding models failed to load.")
#             st.stop()

# embedding_model = load_embedding_model()

# # ------------------- Load CrossEncoder Reranker -------------------
# try:
#     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", use_auth_token=HF_TOKEN)
# except Exception as e:
#     st.error(f"❌ Failed to load reranker model: {e}")
#     st.stop()

# # ------------------- UI Title & Description -------------------
# st.title("⚖️ LEGAL ASSISTANT")
# st.markdown("This AI-powered legal assistant retrieves relevant legal documents and provides accurate responses to your legal queries.")

# # ------------------- User Input -------------------
# query = st.text_input("Enter your legal question:")

# # ------------------- Generate Answer -------------------
# if st.button("Generate Answer"):
#     if not query:
#         st.warning("Please enter a legal question before generating an answer.")
#         st.stop()

#     if len(query.split()) < 4:
#         st.warning("Your query seems incomplete. Please provide more details.")
#         st.stop()

#     with st.spinner("🔍 Searching..."):
#         try:
#             query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
#         except Exception as e:
#             st.error(f"Failed to embed query: {e}")
#             st.stop()

#         try:
#             search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
#         except Exception as e:
#             st.error(f"Pinecone query failed: {e}")
#             st.stop()

#         if not search_results or "matches" not in search_results or not search_results["matches"]:
#             st.warning("No relevant results found. Try rephrasing your query.")
#             st.stop()

#         context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

#         # Rerank results
#         try:
#             rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
#             ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
#         except Exception as e:
#             st.warning("Reranking failed. Using original order.")
#             ranked_results = [(chunk, 0) for chunk in context_chunks]

#         top_chunks = [r[0] for r in ranked_results[:min(5, len(ranked_results))]]
#         context_text = "\n\n".join(top_chunks)

#         # LLM Prompt
#         prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

# Context:
# {context_text}

# Question: {query}

# Answer:"""

#         # Query Together AI
#         try:
#             response = requests.post(
#                 "https://api.together.xyz/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": "meta-llama/Llama-3-70B-Instruct",
#                     "messages": [
#                         {"role": "system", "content": "You are an expert in legal matters."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "temperature": 0.2
#                 }
#             )
#             answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No valid response from AI.")
#         except Exception as e:
#             st.error(f"⚠️ Failed to get response from Together AI: {e}")
#             st.stop()

#         st.success("AI Response:")
#         st.write(answer)

# # ------------------- Footer -------------------
# st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)




import os
import streamlit as st
import asyncio
from huggingface_hub import login
from pinecone import Pinecone, ServerlessSpec

# Optional: Avoid torch/classes crash from streamlit
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load secrets safely
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", None)
TOGETHER_AI_API_KEY = st.secrets.get("TOGETHER_AI_API_KEY", None)

# Hugging Face login (optional)
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        st.info("🔐 Logged into Hugging Face successfully!")
    except Exception as e:
        st.warning("⚠️ Hugging Face login failed: " + str(e))
else:
    st.warning("⚠️ No Hugging Face token found in secrets — proceeding without login.")

# Initialize Pinecone (updated SDK usage)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Example: Check if index exists or create it
index_name = "my_index"
dimension = 1536

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

# Streamlit UI
st.set_page_config(page_title="⚖️ Legal Assistant", layout="wide")

st.title("⚖️ LEGAL ASSISTANT")
st.markdown("Ask legal questions and get AI-powered answers from legal documents.")

# Ask user a question
user_question = st.text_input("🔎 Ask a legal question:")

if user_question:
    with st.spinner("Thinking..."):
        # Replace this with your actual response generation logic
        st.success(f"✅ You asked: **{user_question}**\n\n📄 *Response will go here.*")


