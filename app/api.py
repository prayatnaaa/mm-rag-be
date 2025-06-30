import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.loader.youtube_loader import load_youtube_data
from app.retriever.chromadb_index import search
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources, delete_all_sources
)
from typing import Optional
from app.agent_executor import run_agentic_rag
from PIL import Image

router = APIRouter()
os.makedirs("storage", exist_ok=True)

@router.post("/source/youtube")
def add_youtube(url: str = Form(...)):
    return load_youtube_data(url)

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

@router.delete("/source")
def remove_all_sources():
    success = delete_all_sources()
    if not success:
        raise HTTPException(status_code=500, detail="No Source Available")
    return {"status": "deleted all sources"}

@router.patch("/source/{source_id}")
def toggle_source(source_id: str, active: bool = Form(...)):
    success = set_active_status(source_id, active)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "updated", "active": active}

# @router.post("/test-query")
# async def test_query(question: str = Form(...)):
#     """
#     Test query endpoint for debugging purposes.
#     """
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty")
    
#     search_text = search(question, n_results=5, modality="text")
#     search_image = search(question, n_results=5, modality="image")

#     return {
#         "text_results": search_text,
#         "image_results": search_image
#     }

@router.post("/ask")
async def ask(    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Accepts a question and an optional image, returns an answer using the agent executor.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    image_path = None
    if image:
        image_path = f"/tmp/{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

    return search(question, image=image_path, n_results=5)

@router.post("/rag")
async def rag_pipeline(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
 
    if not question and not image:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    pil_image = None

    if image:
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Invalid image format. Must be path or PIL.Image.")

    return run_agentic_rag(question, image=pil_image, n_chunks=15)