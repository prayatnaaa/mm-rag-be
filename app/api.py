import os
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from typing import Optional
from PIL import Image
import io
from app.loader.youtube_loader import load_youtube_data
from app.loader.pdf_loader import load_pdf_data
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources, delete_all_sources
)
from app.datastore.model import ChatCreateRequest, ChatCreateResponse, ChatHistoryResponse, create_new_chat, get_chat_history
from app.agent_executor import run_agentic_rag
from app.retriever.chromadb_index import search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
os.makedirs("storage", exist_ok=True)

async def process_youtube_with_logging(url: str):
    """
    Process YouTube video with logging.
    """
    try:
        logger.info(f"Starting YouTube processing for URL: {url}")
        result = load_youtube_data(url)
        logger.info(f"Completed YouTube processing for URL: {url}")
        return result
    except Exception as e:
        logger.error(f"Failed to process YouTube URL {url}: {str(e)}")
        raise

@router.post("/source/youtube")
async def add_youtube(background_tasks: BackgroundTasks, url: str = Form(...)):
    """
    Adds a YouTube video for processing in the background.
    """
    if not url.startswith(("https://www.youtube.com/", "https://youtu.be/")):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    background_tasks.add_task(process_youtube_with_logging, url)
    return {
        "status": "processing",
        "message": "YouTube video is being processed in the background",
        "url": url
    }

@router.post("/source/pdf")
async def add_pdf(
    file: UploadFile = File(...),
    source_id: Optional[str] = Form(None)
):
    """
    Adds a PDF file for processing.
    If source_id is provided, it will be used; otherwise, a new one will be generated.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    contents = await file.read()
    pdf_path = os.path.join("storage", file.filename)
    
    with open(pdf_path, "wb") as f:
        f.write(contents)

    if not source_id:
        source_id = f"pdf_{file.filename.split('.')[0]}"

    try:
        result = load_pdf_data(pdf_path, source_id)
        return result
    except Exception as e:
        logger.error(f"Failed to process PDF {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.get("/source/list")
def list_all():
    """
    Lists all sources in the database.
    """
    return list_sources()

@router.get("/source/active")
def list_active():
    """
    Lists all active sources in the database.
    """
    return get_active_sources()

@router.delete("/source/{source_id}")
def remove_source(source_id: str):
    """
    Deletes a specific source by ID.
    """
    success = delete_source(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "deleted"}

@router.delete("/source")
def remove_all_sources():
    """
    Deletes all sources from the database.
    """
    success = delete_all_sources()
    if not success:
        raise HTTPException(status_code=500, detail="No sources available")
    return {"status": "deleted all sources"}

@router.patch("/source/{source_id}")
def toggle_source(source_id: str, active: bool = Form(...)):
    """
    Toggles the active status of a source.
    """
    success = set_active_status(source_id, active)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "updated", "active": active}

@router.post("/rag")
async def rag_pipeline(
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Processes a query with optional image input using the multimodal RAG agent.
    Returns an answer with contexts, including YouTube URLs and timestamps.
    """
    if not question and not image:
        raise HTTPException(status_code=400, detail="At least one of question or image must be provided")

    pil_image = None
    if image:
        try:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    try:
        result = run_agentic_rag(query=question or "", image=pil_image)
        
        if "error" in result["answer"].lower():
            raise HTTPException(status_code=500, detail=result["answer"])

        return {
            "query": result["query"],
            "answer": result["answer"],
            "contexts": [
                {
                    "text": ctx["text"],
                    "source_id": ctx["metadata"].get("source_id"),
                    "youtube_url": ctx["metadata"].get("youtube_url", "unknown"),
                    "start_time": ctx["metadata"].get("start_time"),
                    "end_time": ctx["metadata"].get("end_time"),
                    "image_urls": ctx["metadata"].get("image_urls", "[]"),
                    "distance": ctx["distance"]
                } for ctx in result.get("contexts", [])
            ]
        }
    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

@router.post("/ask")
async def test_search(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Test the search function directly with a question and optional image.
    """
    pil_image = None
    if image:
        try:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    try:
        result = search(
            query=question,
            image=pil_image,
            n_results=5
        )

        if not result["results"]["documents"][0]:
            raise HTTPException(status_code=404, detail="No results found for the query.")
        
        return {
            "query": result["query"],
            "results": [
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                } for doc, meta, dist in zip(
                    result["results"]["documents"][0],
                    result["results"]["metadatas"][0],
                    result["results"]["distances"][0]
                )
            ]
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")