import os


def ensure_directories():
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("storage/faiss_index", exist_ok=True)


def save_uploaded_file(file, upload_dir: str) -> str:
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path
