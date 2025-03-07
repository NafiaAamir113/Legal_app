# import streamlit as st
# import pinecone
# from sentence_transformers import SentenceTransformer

# # 🔹 Load Pinecone API key and environment
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# PINECONE_ENV = st.secrets["PINECONE_ENV"]
# INDEX_NAME = "legaldata-index"

# # 🔹 Initialize Pinecone client
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# # 🔹 Connect to the index
# if INDEX_NAME not in [index_info['name'] for index_info in pc.list_indexes()]:
#     st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
#     st.stop()

# index = pc.Index(INDEX_NAME)
# st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# # 🔹 Load Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # 🔹 Streamlit UI
# st.title("🔎 Legal Document Search")
# query = st.text_input("Enter your legal query:")

# if st.button("Search") and query:
#     query_vector = model.encode(query).tolist()
#     results = index.query(vector=query_vector, top_k=5, include_metadata=True)

#     if results and "matches" in results:
#         for match in results["matches"]:
#             st.write(f"**Score:** {match['score']}")
#             st.write(f"**Document:** {match['metadata'].get('text', 'No content available')}")
#             st.write("---")
#     else:
#         st.warning("⚠️ No relevant documents found.")


import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import requests  

# 🔹 Load API Keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TOGETHER_AI_API_KEY = st.secrets.get("TOGETHER_AI_API_KEY")  

INDEX_NAME = "legaldata-index"

# 🔹 Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Check if Index Exists
if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

index = pc.Index(INDEX_NAME)
st.success(f"✅ Connected to '{INDEX_NAME}'")

# 🔹 Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Expand Legal Keywords
def expand_legal_terms(query):
    synonyms = {
        "contract": ["agreement", "deal", "obligation"],
        "fraud": ["misrepresentation", "deception"],
        "plaintiff": ["claimant", "complainant"],
        "defendant": ["accused", "respondent"]
    }
    words = query.split()
    expanded_query = [word for word in words]  
    for word in words:
        if word.lower() in synonyms:
            expanded_query.extend(synonyms[word.lower()])
    return " ".join(expanded_query)

# 🔹 AI Model to Generate a Direct Legal Answer
def generate_legal_answer(query, case_text):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": "You are an expert legal assistant. Provide a clear and concise legal answer."},
            {"role": "user", "content": f"Question: {query}\n\nRelevant Legal Case:\n{case_text}"}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "⚠️ Unable to generate a response. Try again."

# 🔹 Streamlit UI
st.title("🔎 Legal Assistant")

# 🔹 User Input
query = st.text_input("Enter your legal question:")

if st.button("Search") and query:
    with st.spinner("🔍 Searching legal cases..."):
        
        query = query.lower().strip()
        expanded_query = expand_legal_terms(query)
        query_vector = model.encode(expanded_query).tolist()

        # 🔹 Retrieve from Pinecone
        try:
            results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        except Exception as e:
            st.error(f"⚠️ Pinecone Query Error: {str(e)}")
            st.stop()

        # 🔹 Extract Relevant Cases
        legal_cases = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            case_text = metadata.get("full_text", "")
            if case_text:
                legal_cases.append(case_text)

        # 🔹 Generate Answer
        if legal_cases:
            combined_text = "\n\n".join(legal_cases[:10])  # Use top 2 cases
            legal_answer = generate_legal_answer(query, combined_text)
            st.subheader("📌 Legal Answer:")
            st.write(legal_answer)
        else:
            st.warning("⚠️ No relevant cases found. Try modifying your question.")
