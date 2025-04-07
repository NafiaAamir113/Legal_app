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



import streamlit as st
import requests
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    st.error("Unable to import Pinecone SDK. Please notify the developer.")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="⚖️ LEGAL ASSISTANT", layout="wide")

# Load secrets
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]
except Exception:
    st.error("API keys not configured. Please notify the developer.")
    st.stop()

# Pinecone setup
INDEX_NAME = "lawdata-index"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes]

    if INDEX_NAME not in index_names:
        st.error(f"⚠️ The index '{INDEX_NAME}' is not available. Please notify the developer.")
        st.stop()

    index = pc.Index(INDEX_NAME)

except Exception:
    st.error("⚠️ Failed to connect to Pinecone. Please notify the developer.")
    st.stop()

# Load embedding models
try:
    embedding_model = SentenceTransformer("BAAI/bge-large-en")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    st.error("⚠️ Failed to load embedding models. Please notify the developer.")
    st.stop()

# Page Title and Description
st.title("⚖️ LEGAL ASSISTANT")
st.markdown("Ask legal questions and get AI-powered answers from legal documents.")

# User Input
query = st.text_input("Enter your legal question:")

# Generate Answer
if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a legal question before generating an answer.")
        st.stop()

    if len(query.split()) < 4:
        st.warning("Your question seems too short. Try adding more details.")
        st.stop()

    with st.spinner("🔍 Searching legal knowledge base..."):
        try:
            query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
            search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        except Exception:
            st.error("⚠️ An error occurred while querying legal documents. Please notify the developer.")
            st.stop()

        matches = search_results.get("matches", [])
        if not matches:
            st.warning("No relevant results found. Try rephrasing your query.")
            st.stop()

        # Extract and rerank
        try:
            context_chunks = [match["metadata"]["text"] for match in matches]
            rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
            ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
            context_text = "\n\n".join([r[0] for r in ranked_results[:min(5, len(ranked_results))]])
        except Exception:
            st.error("⚠️ Failed during reranking or context generation. Please notify the developer.")
            st.stop()

        # LLM Prompt
        prompt = f"""
        You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

        Context:
        {context_text}

        Question: {query}

        Answer:"""

        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/Llama-3-70B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are an expert in legal matters."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No valid response from AI.")
            st.success("✅ AI Response:")
            st.write(answer)
        except Exception:
            st.error("⚠️ AI model could not generate a response. Please notify the developer.")

# Footer
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit | Please contact the developer for any issues</p>", unsafe_allow_html=True)


# # ------------------- Footer -------------------
# st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)






