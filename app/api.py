import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.loader.youtube_loader import load_youtube_data
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources, delete_all_sources
)
from app.agent_executor import AgentExecutor
from typing import Optional

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
