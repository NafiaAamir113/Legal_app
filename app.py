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
#             st.stop()

#         # Extract text chunks from results
#         context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

#         # Rerank results
#         rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
#         ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

#         # Select dynamic number of chunks (min available or 5)
#         num_chunks = min(len(ranked_results), 5)
#         context_text = "\n\n".join([r[0] for r in ranked_results[:num_chunks]])

#         # Construct LLM prompt
#         prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

#         Context:
#         {context_text}

#         Question: {query}

#         Answer:"""

#         # Query Together AI
#         response = requests.post(
#             "https://api.together.xyz/v1/chat/completions",
#             headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
#             json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                   "messages": [{"role": "system", "content": "You are an expert in legal matters."},
#                                {"role": "user", "content": prompt}], "temperature": 0.2}
#         )

#         answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No valid response from AI.")
#         st.success("AI Response:")
#         st.write(answer)

# # Footer with emoji
# st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)


import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit page setup
st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-index"
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        st.error(f"Index '{INDEX_NAME}' not found.")
        st.stop()
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Load embedding and reranking models
@st.cache_resource
def load_models():
    return (SentenceTransformer("BAAI/bge-large-en"), 
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))

embedding_model, reranker = load_models()

# Page Title
st.title("⚖️ LEGAL ASSISTANT")

# Short App Description
st.markdown("This AI-powered legal assistant retrieves relevant legal documents and provides accurate responses to your legal queries.")

# Input field
query = st.text_input("Enter your legal question:")

# Generate Answer Button
if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a legal question before generating an answer.")
        st.stop()

    # Check for incomplete query
    if len(query.split()) < 4:
        st.warning("Your query seems incomplete. Please provide more details.")
        st.stop()

    with st.spinner("Searching..."):
        try:
            query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
        except Exception as e:
            st.error(f"Failed to generate query embedding: {e}")
            st.stop()

        # Query Pinecone with increased top_k for better retrieval
        try:
            search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.stop()

        # Stop execution if no documents are found
        if not search_results or "matches" not in search_results or not search_results["matches"]:
            st.warning("No relevant legal case found in the database. Please refine your query.")
            st.stop()

        # Extract text chunks and citations from results
        retrieved_cases = []
        case_citations = []  # Stores case names for reference in AI response
        for match in search_results["matches"]:
            if "text" in match["metadata"]:
                case_text = match["metadata"]["text"]
                case_source = match["metadata"].get("source", "Unknown Case")  # Adds citation if available
                retrieved_cases.append(f"📜 **[{case_source}]**\n{case_text}")
                case_citations.append(f"[{case_source}]")  # Save for AI reference

        # Rerank results if more than one retrieved
        if len(retrieved_cases) > 1:
            rerank_scores = reranker.predict([(query, chunk) for chunk in retrieved_cases])
            ranked_results = sorted(zip(retrieved_cases, rerank_scores), key=lambda x: x[1], reverse=True)
        else:
            ranked_results = [(chunk, 1.0) for chunk in retrieved_cases]

        # Select top 5 case texts
        num_chunks = min(len(ranked_results), 5)
        context_text = "\n\n".join([r[0] for r in ranked_results[:num_chunks]])

        # 🔥 Improved LLM prompt to force citation inclusion
        prompt = f"""You are a legal assistant. Your job is to provide detailed answers from the retrieved legal cases and provide a response based strictly on the given context.

        Context:
        {context_text}

        Question: {query}

        Answer:
        - Ensure that each key legal point references a retrieved case.
        - If a legal point is derived from a specific case, mention the case in brackets (e.g., [1969 SCMR 564]).
        - If the retrieved cases do not contain a relevant answer, state: 'No relevant case found in the database.'"""

        # Query Together AI
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                      "messages": [{"role": "system", "content": "You are an expert in legal matters."},
                                   {"role": "user", "content": prompt}], "temperature": 0.2}
            )

            response_data = response.json()
            answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not answer or "No relevant case found" in answer:
                answer = "No relevant case found in the database."

        except Exception as e:
            st.error(f"AI query failed: {e}")
            st.stop()

        # Display results
        st.success("AI Response:")
        st.write(answer)

        # Show referenced cases
        if case_citations:
            st.markdown("### 📌 Referenced Cases:")
            st.markdown(", ".join(case_citations))

# Footer
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)





