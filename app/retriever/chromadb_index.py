import chromadb
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np
from deep_translator import GoogleTranslator
from typing import Optional, Dict, Union
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32, device_map={"": "cpu"})
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

MAX_TOKENS = 72

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="embeddings", embedding_function=None)

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return text

def embed_text(text: str, truncate=True) -> np.ndarray:
    text = translate_to_english(text)

    if truncate:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
            text = tokenizer.convert_tokens_to_string(tokens)

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).squeeze().numpy()
    return normalize(embedding)

def embed_image(image: Union[str, Image.Image]) -> np.ndarray:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze().numpy()
    return normalize(embedding)

def add_embedding(vec: np.ndarray, metadata: dict) -> str:
    if "source_id" not in metadata and "video_id" in metadata:
        metadata["source_id"] = f"yt_{metadata['video_id']}"

    doc_id = f"{metadata['modality']}_{metadata['source_id']}_{int(metadata['start_time'] * 1000)}"

    try:
        collection.add(
            documents=[metadata.get("text", "(image)")],
            embeddings=[vec.tolist()],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return doc_id
    except Exception as e:
        logger.error(f"Error adding embedding to ChromaDB: {str(e)}")
        raise

def add_embedding_pdf(vec: np.ndarray, metadata: dict) -> str:
    if "source_id" not in metadata and "video_id" in metadata:
        metadata["source_id"] = f"yt_{metadata['video_id']}"

    modality = metadata.get("modality", "unknown")
    source_id = metadata.get("source_id", "nosrc")
    
    if "video_id" in metadata and "start_time" in metadata:
        unique_id = f"{metadata['video_id']}_{int(metadata['start_time'] * 1000)}"
    elif "page" in metadata:
        unique_id = f"page_{metadata['page']}_{uuid.uuid4().hex[:8]}"
    else:
        unique_id = uuid.uuid4().hex  

    doc_id = f"{modality}_{source_id}_{unique_id}"

    content = metadata.get("text", "(image)")

    try:
        collection.add(
            documents=[content],
            embeddings=[vec.tolist()],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return doc_id
    except Exception as e:
        logger.error(f"Error adding PDF embedding to ChromaDB: {str(e)}")
        raise

def search(
    query: Optional[str] = None,
    image: Optional[Union[str, Image.Image]] = None,
    n_results: int = 5,
    where: Optional[Dict] = None,
    text_weight: float = 0.5
) -> Dict:
    
    if not query and not image:
        raise ValueError("You must provide at least a text query or an image.")

    # Initialize where clause
    where_conditions = [{"active": {"$eq": True}}]
    
    # Add user-provided where conditions
    if where:
        where_conditions.append(where)

    # Handle image-only query
    if image and not query:
        where_conditions.append({"modality": {"$eq": "image"}})
        where_clause = {"$and": where_conditions}
        
        image = Image.open(image).convert("RGB") if isinstance(image, str) else image
        image_embedding = embed_image(image)
        
        result = collection.query(
            query_embeddings=[image_embedding.tolist()],
            n_results=n_results,
            where=where_clause
        )
        
        # Log distances for debugging
        distances = result["distances"][0]
        logger.info(f"Image-only query distances: {distances}")
        
        return {
            "query": "(image)",
            "results": result
        }

    # Handle text-only query
    if query and not image:
        where_conditions.append({"modality": {"$eq": "multimodal"}})
        where_clause = {"$and": where_conditions}
        
        query = translate_to_english(query)
        if not query.strip():
            raise ValueError("Query text is empty after translation.")
        
        text_embedding = embed_text(query)
        
        result = collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results,
            where=where_clause
        )
        
        return {
            "query": query,
            "results": result
        }

    # Handle multimodal query (text + image)
    where_conditions.append({"modality": {"$eq": "multimodal"}})  # Prioritize text modality for multimodal
    where_clause = {"$and": where_conditions}
    
    query = translate_to_english(query)
    if not query.strip():
        raise ValueError("Query text is empty after translation.")

    vec_text = embed_text(query)
    image = Image.open(image).convert("RGB") if isinstance(image, str) else image
    vec_image = embed_image(image)
    
    # Weighted average for multimodal embedding
    combined_embedding = (text_weight * vec_text + (1 - text_weight) * vec_image)
    combined_embedding = normalize(combined_embedding)

    result = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=n_results,
        where=where_clause
    )
    
    # Log distances for debugging
    distances = result["distances"][0]
    logger.info(f"Multimodal query distances: {distances}")
    
    return {
        "query": query,
        "results": result
    }