import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from typing import Optional
from PIL import Image
import io
from app.loader.youtube_loader import load_youtube_data
from app.db.metadata_store import (
    list_sources, delete_source, set_active_status, get_active_sources, delete_all_sources
)
from app.datastore.model import ChatCreateRequest, ChatCreateResponse, ChatHistoryResponse, create_new_chat, get_chat_history
from app.agent_executor import run_agentic_rag
from app.retriever.chromadb_index import search

router = APIRouter()
os.makedirs("storage", exist_ok=True)

# @router.post("/source/youtube")
# def add_youtube(url: str = Form(...)):
#     return load_youtube_data(url)

@router.post("/source/youtube")
async def add_youtube(background_tasks: BackgroundTasks, url: str = Form(...)):
    """
    Adds a YouTube video for processing in the background.
    """
    if not url.startswith("https://www.youtube.com/"):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    background_tasks.add_task(load_youtube_data, url)
    return {
        "status": "processing",
        "message": "YouTube video is being processed in the background",
        "url": url
    }

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
    Returns an answer with separated text and image contexts, including YouTube URLs.
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

    # Run the agentic RAG pipeline
    result = run_agentic_rag(query=question or "", image=pil_image)
    
    if "error" in result["answer"].lower() or "kesalahan" in result["answer"].lower():
        raise HTTPException(status_code=500, detail=result["answer"])

    return {
        "query": result["query"],
        "answer": result["answer"],
        "text_contexts": [
            {
                "text": ctx["text"],
                "source_id": ctx["metadata"].get("source_id"),
                "youtube_url": ctx["metadata"].get("youtube_url", "unknown"),
                "start_time": ctx["metadata"].get("start_time"),
                "end_time": ctx["metadata"].get("end_time"),
                "distance": ctx["distance"]
            } for ctx in result.get("text_contexts", [])
        ],
        "image_contexts": [
            {
                "image_url": ctx["metadata"].get("image_url"),
                "source_id": ctx["metadata"].get("source_id"),
                "youtube_url": ctx["metadata"].get("youtube_url", "unknown"),
                "timestamp": ctx["metadata"].get("start_time"),
                "distance": ctx["distance"]
            } for ctx in result.get("image_contexts", [])
        ]
    }

@router.post("/chats", response_model=ChatCreateResponse, status_code=201)
def create_chat_endpoint(request: ChatCreateRequest):
    """
    Creates a new chat session with a given topic.
    """
    chat_id = create_new_chat(topic=request.topic)
    
    if chat_id is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to create a new chat session in the database."
        )
        
    return {"chat_id": chat_id, "topic": request.topic}

@router.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
def get_chat_endpoint(chat_id: int):
    """
    Retrieves the full history of a specific chat session, including messages and sources.
    """
    history = get_chat_history(chat_id=chat_id)
    
    if history is None:
        raise HTTPException(
            status_code=404,
            detail=f"Chat with ID {chat_id} not found."
        )
        
    return history

@router.post("/ask")
def test_search(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
  
  result = search(
      query=question,
      image=image,
      n_results=5,
      where={"active": {"$eq": True}}
  )

  if result["text_results"]["documents"] == [[]] and result["image_results"]["documents"] == [[]]:
      raise HTTPException(status_code=404, detail="No results found for the query.")
  
  return result