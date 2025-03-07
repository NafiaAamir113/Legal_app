import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

# 🔹 Load Pinecone API key and environment
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = "legaldata-index"

# 🔹 Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Connect to the index
if INDEX_NAME not in [index_info['name'] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

index = pc.Index(INDEX_NAME)
st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# 🔹 Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Streamlit UI
st.title("🔎 Legal Document Search")
query = st.text_input("Enter your legal query:")

if st.button("Search") and query:
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    if results and "matches" in results:
        for match in results["matches"]:
            st.write(f"**Score:** {match['score']}")
            st.write(f"**Document:** {match['metadata'].get('text', 'No content available')}")
            st.write("---")
    else:
        st.warning("⚠️ No relevant documents found.")
