import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.loader.youtube_loader import load_youtube_data
from app.retriever.chromadb_index import embed_text, embed_image, search
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources, delete_all_sources
)
from app.agent_executor import AgentExecutor
from typing import Optional
import io
from PIL import Image
import numpy as np
from app.retriever.chromadb_index import collection, normalize

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

@router.post("/query")
async def query(
    question: str = Form(...), 
    image: Optional[UploadFile] = File(None)
):
    image_path = None
    if image:
        image_path = f"/tmp/{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

    return AgentExecutor(question, image_path=image_path)

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

@router.post("/test-query")
async def test_query(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Accepts text, image, or both, and returns separate search results
    for text and image collections.
    """
    if not question and not image:
        raise HTTPException(status_code=400, detail="Provide either text or image input.")

    query_embeddings = []

    if question:
        text_embedding = embed_text(question)
        query_embeddings.append(text_embedding)

    if image:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_embedding = embed_image(pil_image)
        query_embeddings.append(image_embedding)

    # If both text and image were provided, average them for a hybrid search
    if query_embeddings:
        combined_embedding = normalize(np.mean(query_embeddings, axis=0))
    else:
        raise HTTPException(status_code=400, detail="Failed to compute embedding.")

    # Search separately in both modalities
    text_results = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=5,
        where={"modality": "text"}
    )

    image_results = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=5,
        where={"modality": "image"}
    )

    return {
        "text_results": text_results,
        "image_results": image_results
    }