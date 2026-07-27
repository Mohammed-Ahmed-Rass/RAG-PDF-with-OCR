from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import UPLOAD_DIR
from backend.rag_service import ask_with_rag, ask_without_rag, ingest_pdf
from backend.schemas import AskRequest, AskResponse
from backend.utils import ensure_directories, save_uploaded_file

app = FastAPI(title="RAG PDF Teaching Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    ensure_directories()


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_path = save_uploaded_file(file, UPLOAD_DIR)
    result = ingest_pdf(file_path)
    return {
        "filename": file.filename,
        **result,
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    no_rag_answer = ask_without_rag(request.question)
    rag_result = ask_with_rag(request.question)
    return {
        "question": request.question,
        "no_rag_answer": no_rag_answer,
        "rag_answer": rag_result["answer"],
        "retrieved_chunks": rag_result["chunks"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
