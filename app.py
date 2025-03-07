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
import requests  # For AI-generated summaries (Together AI)

# 🔹 Load API Keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TOGETHER_AI_API_KEY = st.secrets.get("TOGETHER_AI_API_KEY")  # For AI summaries

INDEX_NAME = "legaldata-index"

# 🔹 Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Check if Index Exists
if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    st.error(f"❌ Index '{INDEX_NAME}' not found. Check your Pinecone dashboard.")
    st.stop()

index = pc.Index(INDEX_NAME)
st.success(f"✅ Successfully connected to '{INDEX_NAME}'")

# 🔹 Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Function to Expand Legal Keywords (Optional)
def expand_legal_terms(query):
    synonyms = {
        "contract": ["agreement", "deal", "obligation"],
        "fraud": ["misrepresentation", "deception"],
        "plaintiff": ["claimant", "complainant"],
        "defendant": ["accused", "respondent"]
    }
    words = query.split()
    expanded_query = []
    for word in words:
        expanded_query.append(word)
        if word.lower() in synonyms:
            expanded_query.extend(synonyms[word.lower()])
    return " ".join(expanded_query)

# 🔹 Function to Get AI Summary Using Llama-3.3-70B
def generate_llama_summary(summary_prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": summary_prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json()}"

# 🔹 Streamlit UI
st.title("🔎 Legal Case Retrieval System")

# 🔹 User Input
query = st.text_input("Enter your legal query:")

if st.button("Search") and query:
    with st.spinner("🔍 Searching legal cases..."):
        
        # 🔹 Query Processing
        query = query.lower().strip()  # Normalize query
        expanded_query = expand_legal_terms(query)  # Expand keywords
        query_vector = model.encode(expanded_query).tolist()

        # 🔹 Retrieve from Pinecone
        try:
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"⚠️ Pinecone Query Error: {str(e)}")
            st.stop()

        # 🔹 Filter & Rank Results
        relevant_cases = []
        for match in results.get("matches", []):
            score = match.get("score", 0)
            metadata = match.get("metadata", {})
            case_name = metadata.get("case_name", "Unknown Case")
            citation = metadata.get("citation", "No citation available")
            summary = metadata.get("summary", "No summary provided")
            case_type = metadata.get("case_type", "Unknown")

            # Filter by Relevance
            if score > 0.60:  # Set similarity threshold
                relevant_cases.append((score, case_name, citation, summary, case_type))

        # 🔹 Display Results
        if relevant_cases:
            st.success("✅ Here are the most relevant legal cases:")

            for i, (score, case_name, citation, summary, case_type) in enumerate(relevant_cases, 1):
                st.write(f"**{i}. {case_name}**")
                st.write(f"🔹 **Score:** {round(score, 2)}")
                st.write(f"🔹 **Citation:** {citation}")
                st.write(f"🔹 **Case Type:** {case_type}")
                st.write(f"📄 **Summary:** {summary}")
                st.write("---")
        
            # 🔹 AI Summary of Retrieved Cases (Using Llama-3.3-70B)
            if TOGETHER_AI_API_KEY:
                if st.button("Generate AI Summary"):
                    st.subheader("📌 AI-Generated Case Summary:")
                    summary_prompt = "Summarize these legal cases in simple terms:\n\n"
                    for _, case_name, _, summary, _ in relevant_cases:
                        summary_prompt += f"- {case_name}: {summary}\n"
                    
                    llama_summary = generate_llama_summary(summary_prompt)
                    st.write(llama_summary)
        
        else:
            st.warning("⚠️ No relevant cases found. Try modifying your query.")

