import streamlit as st
import pinecone  # Correct import for Pinecone
from sentence_transformers import SentenceTransformer

# 🔹 Load Pinecone API key and environment
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENV = st.secrets["PINECONE_ENV"]
    INDEX_NAME = "legaldata-index"
except KeyError:
    st.error("❌ Pinecone API key or environment is missing in Streamlit secrets.")
    st.stop()

# 🔹 Initialize Pinecone
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"❌ Error initializing Pinecone: {e}")
    st.stop()

# 🔹 Check available indexes
existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
st.write("✅ Available Indexes:", existing_indexes)

# 🔹 Ensure the index exists
if INDEX_NAME not in existing_indexes:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

# 🔹 Connect to the index
try:
    index = pc.Index(INDEX_NAME)
    st.success(f"✅ Successfully connected to '{INDEX_NAME}'")
except Exception as e:
    st.error(f"❌ Failed to connect to index: {e}")
    st.stop()

# 🔹 Load Sentence Transformer model
st.write("🔄 Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
st.success("✅ Model loaded successfully.")

# 🔹 Streamlit UI
st.set_page_config(page_title="Legal Document Search", page_icon="🔍", layout="wide")
st.title("🔎 Legal Document Search")
st.write("Enter a legal query to retrieve relevant documents from the database.")

# 🔹 User input
query = st.text_input("Enter your legal query:")

if st.button("Search") and query:
    # Convert query to vector
    query_vector = model.encode(query).tolist()
    
    # Search Pinecone
    try:
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        
        # Display results
        if results and "matches" in results:
            for match in results["matches"]:
                st.write(f"**Score:** {match['score']}")
                st.write(f"**Document:** {match['metadata'].get('text', 'No content available')}")
                st.write("---")
        else:
            st.warning("⚠️ No relevant documents found.")

    except Exception as e:
        st.error(f"❌ Pinecone search failed: {e}")
