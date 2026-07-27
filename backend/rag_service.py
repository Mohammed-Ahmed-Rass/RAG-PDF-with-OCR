import os
from typing import Any, Dict

import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    OCR_DPI,
    OCR_LANGUAGE,
    OCR_MIN_TEXT_LENGTH,
    POPPLER_PATH,
    TEMPERATURE,
    TOP_K,
)

load_dotenv()

vector_store = None


def _is_scanned_page(text: str) -> bool:
    """A page is treated as scanned/image-based if PyPDFLoader extracted
    little or no real text from it."""
    return len(text.strip()) < OCR_MIN_TEXT_LENGTH


def _ocr_page(file_path: str, page_number: int) -> str:
    """Render a single PDF page to an image (via pdf2image/Poppler) and
    run Tesseract OCR (via pytesseract) on it, returning the recognized text."""
    images = convert_from_path(
        file_path,
        dpi=OCR_DPI,
        first_page=page_number + 1,
        last_page=page_number + 1,
        poppler_path=POPPLER_PATH,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=OCR_LANGUAGE)


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        if _is_scanned_page(doc.page_content):
             page_number = doc.metadata.get("page", 0)
             doc.page_content = _ocr_page(file_path, page_number)

             print("=" * 60)
             print(f"OCR Page {page_number + 1}")
             print(doc.page_content[:500])
             print("=" * 60)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks):
    global vector_store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def ingest_pdf(file_path: str) -> Dict[str, Any]:
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    build_vector_store(chunks)
    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "message": "PDF indexed successfully",
    }


def get_retriever():
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": TOP_K,
            "score_threshold": 0.1,
        },
    )


def build_rag_prompt(question: str, retrieved_docs):
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    return f"""You are a document-grounded assistant.
Answer the question using only the context below.
If the answer is not available in the context, say:
"I don't know based on the provided PDF."

Context:
{context}

Question:
{question}

Answer:"""


def ask_without_rag(question: str) -> str:
    llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=1000,
     )
    response = llm.invoke(question)
    return response.content


def ask_with_rag(question: str) -> Dict[str, Any]:
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs_and_scores = vector_store.similarity_search_with_score(question, k=TOP_K)
    retrieved_docs = [doc for doc, score in docs_and_scores]
    prompt = build_rag_prompt(question, retrieved_docs)

    llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=1000,
     )
    response = llm.invoke(prompt)

    chunks = []
    for doc, score in docs_and_scores:
        source = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", 0) + 1
        chunks.append(
            {
                "content": doc.page_content,
                "source": source,
                "page": page,
                "score": float(score),
            }
        )

    return {
        "answer": response.content,
        "chunks": chunks,
    }
