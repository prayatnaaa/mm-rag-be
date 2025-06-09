import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.loader.pdf_loader import (load_pdf_data, load_pdf_data_from_minio)
from app.loader.txt_loader import (load_txt_data, load_txt_data_from_minio)
from app.loader.youtube_loader import load_youtube_data
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources
)
from app.rag_pipeline import run_rag_pipeline
from app.utils.minio_client import upload_bytes_to_minio
from app.utils.file_utils import generate_source_id

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

@router.get("/source/active")
def list_active():
    return get_active_sources()

@router.delete("/source/{source_id}")
def remove_source(source_id: str):
    success = delete_source(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "deleted"}

@router.patch("/source/{source_id}")
def toggle_source(source_id: str, active: bool = Form(...)):
    success = set_active_status(source_id, active)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "updated", "active": active}

@router.post("/query")
def query(req: dict):
    return run_rag_pipeline(req["question"])

@router.post("/source/upload")
def upload_source(file: UploadFile = File(...), source_id: str = Form(None)):
    if source_id is None:
        source_id = generate_source_id(file.filename)

    ext = file.filename.split(".")[-1]
    file_bytes = file.file.read()

    if ext == "pdf":
        object_name = f"pdfs/{source_id}_{file.filename}"
        upload_bytes_to_minio(file_bytes, object_name, "application/pdf")
        return load_pdf_data_from_minio(object_name, source_id)

    elif ext == "txt":
        object_name = f"txts/{source_id}_{file.filename}"
        upload_bytes_to_minio(file_bytes, object_name, "text/plain")
        return load_txt_data_from_minio(object_name, source_id)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
