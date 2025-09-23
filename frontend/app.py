import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="GenAI RAG Platform", layout="wide")
st.title("📑 GenAI Document Intelligence Platform")
st.write("Upload PDFs and ask questions. Powered by GPT + RAG strategies.")

# ---------------- Upload Section ----------------
with st.expander("📂 Upload PDFs", expanded=True):
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        if response.status_code == 200:
            st.success(response.json().get("message"))
        else:
            st.error(f"Upload failed: {response.text}")

# ---------------- Query Section ----------------
st.subheader("🔎 Ask a Question")

col1, col2 = st.columns(2)
with col1:
    llm_model = st.selectbox("Select LLM Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"])
with col2:
    strategy = st.selectbox("Select RAG Strategy", ["vanilla", "reranking", "hyde"])

query = st.text_input("Enter your question here:")

if st.button("Get Answer") and query:
    try:
        payload = {"query": query, "llm_model": llm_model, "strategy": strategy}
        response = requests.post(f"{BACKEND_URL}/query", json=payload)

        if response.status_code == 200:
            data = response.json()

            # ---------------- Answer ----------------
            st.subheader("💡 Answer")
            st.write(data.get("answer", "No answer returned."))

            # ---------------- Retrieval Metrics ----------------
            st.subheader("📊 Retrieval Metrics")
            metrics = data.get("metrics", {})

            # Show key metrics in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Latency (s)", metrics.get("latency"))
            col2.metric("Chunks Retrieved", metrics.get("chunks_retrieved"))
            col3.metric("Recall@k", metrics.get("recall_at_k"))
            col4.metric("Avg Similarity", metrics.get("avg_similarity"))
            col5.metric("Median Similarity", metrics.get("median_similarity"))
            col6.metric("Query Coverage (%)", metrics.get("query_coverage"))

            # Show similarity bar chart
            sim_scores = metrics.get("similarity_scores", [])
            if sim_scores:
                df = pd.DataFrame({
                    "Chunk": [f"Chunk {i+1}" for i in range(len(sim_scores))],
                    "Similarity": sim_scores
                })
                st.bar_chart(df.set_index("Chunk"))

            # ---------------- Source Chunks ----------------
            st.subheader("📚 Source Chunks")
            for i, chunk in enumerate(data.get("sources", []), start=1):
                filename = chunk['metadata'].get('filename', 'unknown')
                chunk_id = chunk['metadata'].get("chunk_id", "?")
                score = round(chunk["score"], 3)
                with st.expander(f"Source {i}: {filename} (chunk {chunk_id}, score={score})"):
                    st.write(chunk["text"][:1000] + "…")

        else:
            st.error(f"Query failed: {response.text}")

    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
