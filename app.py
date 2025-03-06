import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import os

# 🔹 Initialize Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
if not PINECONE_API_KEY:
    raise Exception("Pinecone API key not found. Set it in your environment variables.")

pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = "legaldata-index"

# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# 🔹 Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Streamlit UI
st.set_page_config(page_title="Legal Document Search", page_icon="🔍", layout="wide")
st.title("🔎 Legal Document Search")
st.write("Enter a legal query to retrieve relevant documents from the database.")

# User input
query = st.text_input("Enter your legal query:")

if st.button("Search") and query:
    # Convert query to vector
    query_vector = model.encode(query).tolist()
    
    # Search Pinecone
    results = index.query(query_vector, top_k=5, include_metadata=True)

    # Display results
    if results and "matches" in results:
        for match in results["matches"]:
            st.write(f"**Score:** {match['score']}")
            st.write(f"**Document:** {match['metadata'].get('text', 'No content available')}")
            st.write("---")
    else:
        st.warning("No relevant documents found.")
