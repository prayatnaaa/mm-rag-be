from fastapi import APIRouter, Form
from pydantic import BaseModel
from typing import List, Dict
from app.loader.youtube_loader import load_youtube_data
from app.db.metadata_store import list_sources
from app.rag_pipeline import run_rag_pipeline

router = APIRouter()

@router.post("/source/youtube")
def add_youtube(url: str = Form(...)):
    return load_youtube_data(url)

@router.get("/source/list")
def list_all():
    return list_sources()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query(req: QueryRequest):
    return run_rag_pipeline(req.question)
