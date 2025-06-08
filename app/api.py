import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.loader.pdf_loader import load_pdf_data
from app.loader.txt_loader import load_txt_data
from app.loader.youtube_loader import load_youtube_data
from app.db.metadata_store import list_sources
from app.rag_pipeline import run_rag_pipeline

router = APIRouter()
os.makedirs("storage", exist_ok=True)

@router.post("/source/youtube")
def add_youtube(url: str = Form(...)):
    return load_youtube_data(url)

@router.post("/source/pdf")
def upload_pdf(file: UploadFile = File(...)):
    path = f"storage/{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    from uuid import uuid4
    source_id = f"pdf_{uuid4().hex[:8]}"
    cnt = load_pdf_data(path, source_id)
    return {"status": "ok", "source_id": source_id, "chunks": cnt}

@router.post("/source/txt")
def upload_txt(file: UploadFile = File(...)):
    path = f"storage/{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    from uuid import uuid4
    source_id = f"txt_{uuid4().hex[:8]}"
    cnt = load_txt_data(path, source_id)
    return {"status": "ok", "source_id": source_id, "chunks": cnt}

@router.get("/source/list")
def list_all():
    return list_sources()

@router.post("/query")
def query(req: dict):
    return run_rag_pipeline(req["question"])
