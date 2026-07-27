import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG PDF Teaching Demo", layout="wide")

st.title("RAG PDF Teaching Demo")
st.markdown("This app shows how Retrieval-Augmented Generation works using one PDF.")

# -----------------------------
# Visual Flow
# -----------------------------
st.header("RAG Flow")
st.graphviz_chart(
    """
digraph {
    rankdir=LR;
    PDF -> Chunking;
    Chunking -> Embeddings;
    Embeddings -> FAISS;
    Question -> Retriever;
    FAISS -> Retriever;
    Retriever -> "Top Chunks";
    "Top Chunks" -> Prompt;
    Question -> Prompt;
    Prompt -> LLM;
    LLM -> Answer;
}
"""
)

# -----------------------------
# Upload PDF
# -----------------------------
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Upload one PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Index PDF"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }
        response = requests.post(f"{BACKEND_URL}/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("PDF indexed successfully")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", result["pages"])
            with col2:
                st.metric("Chunks", result["chunks"])
        else:
            st.error("Failed to upload PDF")

# -----------------------------
# Ask Question
# -----------------------------
st.header("2. Ask a Question")
question = st.text_input("Question", placeholder="Ask something about the PDF...")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        response = requests.post(f"{BACKEND_URL}/ask", json={"question": question})

        if response.status_code != 200:
            st.error("Failed to get answer.")
        else:
            data = response.json()

            st.header("3. Comparison: No RAG vs RAG")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Without RAG")
                st.error("The model answers from general knowledge only.")
                st.write(data["no_rag_answer"])

            with col2:
                st.subheader("With RAG")
                st.success("The model answers using retrieved PDF context.")
                st.write(data["rag_answer"])

            st.header("4. Retrieved Chunks")
            for i, chunk in enumerate(data["retrieved_chunks"], start=1):
                with st.expander(
                    f"Chunk {i} | Source: {chunk['source']} | Page: {chunk['page']}"
                ):
                    st.markdown(
                        f"""**Source:** `{chunk['source']}`  
**Page:** `{chunk['page']}`
**Similarity Score:** `{chunk.get('score', 0):.4f}`
---
{chunk['content']}"""
                    )
