UPLOAD_DIR = "data/uploads"
FAISS_INDEX_PATH = "storage/faiss_index"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

# --- OCR settings (used only for scanned/image-based PDF pages) ---
# If PyPDFLoader extracts fewer than this many non-whitespace characters
# for a page, the page is treated as scanned and OCR is run on it instead.
OCR_MIN_TEXT_LENGTH = 20
# Language(s) passed to Tesseract, e.g. "eng" or "eng+ara"
OCR_LANGUAGE = "eng"
# DPI used when rendering PDF pages to images for OCR (higher = more
# accurate but slower)
OCR_DPI = 300
# Optional: set this to your Poppler "bin" folder path if it is not on
# your system PATH (commonly needed on Windows), e.g.:
# POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
POPPLER_PATH = (
    r"C:\Users\mrass\Desktop\BigData\المشروع\poppler-26.02.0\Library\bin"
)