import chromadb
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Union, List
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32, device_map={"": "cpu"})
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

MAX_TOKENS = 72
DEFAULT_LANG = "en"

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="embeddings", embedding_function=None)

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return DEFAULT_LANG

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def truncate_text(text: str) -> str:
    tokens = clip_tokenizer.tokenize(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = clip_tokenizer.convert_tokens_to_string(tokens)
    return text

def embed_text(text: str, truncate: bool = True) -> np.ndarray:
    if truncate:
        text = truncate_text(text.strip())
    with torch.no_grad():
        embedding = text_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return normalize(embedding)

def embed_image(image: Union[str, Image.Image]) -> np.ndarray:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).squeeze().numpy()
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
        logger.error(f"ChromaDB insert error: {str(e)}")
        raise

# def search(
#     query: Optional[str] = None,
#     image: Optional[Union[str, Image.Image]] = None,
#     n_results: int = 5,
#     where: Optional[Dict] = None,
#     text_weight: float = 0.5
# ) -> Dict:
#     if not query and not image:
#         raise ValueError("At least a query or an image must be provided.")

#     where_conditions = [{"active": {"$eq": True}}]
#     if where:
#         where_conditions.append(where)

#     results = []
#     base_where_clause = {"$and": where_conditions}

#     def safe_query(embedding: np.ndarray, modality: str):
#         clause = {"$and": where_conditions + [{"modality": {"$eq": modality}}]}
#         try:
#             res = collection.query(
#                 query_embeddings=[embedding.tolist()],
#                 n_results=n_results,
#                 where=clause
#             )
#             if not isinstance(res, dict):
#                 logger.error(f"Invalid result from collection.query: {res}")
#                 return []
#             return list(zip(res.get("ids", []), res.get("distances", []), res.get("metadatas", [])))
#         except Exception as e:
#             logger.exception(f"Vector DB query failed for modality '{modality}': {str(e)}")
#             return []

#     if query and not image:
#         query = query.strip()
#         if not query:
#             raise ValueError("Empty query string.")

#         lang = detect_language(query)
#         embeddings = [embed_text(query)]

#         if lang != "en":
#             try:
#                 translated = translate_to_english(query)
#                 embeddings.append(embed_text(translated))
#             except Exception as e:
#                 logger.warning(f"Translation failed, falling back to original query: {str(e)}")

#         for emb in embeddings:
#             results.extend(safe_query(emb, "multimodal"))

#     elif image and not query:
#         try:
#             vec_img = embed_image(image)
#             results.extend(safe_query(vec_img, "image"))
#         except Exception as e:
#             logger.exception(f"Image embedding failed: {str(e)}")

#     else:
#         # Multimodal: text + image
#         try:
#             vec_img = embed_image(image)
#         except Exception as e:
#             logger.exception(f"Image embedding failed: {str(e)}")
#             vec_img = None

#         try:
#             vec_texts = [embed_text(query)]
#             lang = detect_language(query)
#             if lang != "en":
#                 translated = translate_to_english(query)
#                 vec_texts.append(embed_text(translated))
#         except Exception as e:
#             logger.exception(f"Text embedding or translation failed: {str(e)}")
#             vec_texts = []

#         if vec_img is not None and vec_texts:
#             for vec_text in vec_texts:
#                 combined = normalize(text_weight * vec_text + (1 - text_weight) * vec_img)
#                 results.extend(safe_query(combined, "multimodal"))

#     # Deduplicate and sort
#     seen = set()
#     unique_results = []
#     for r in sorted(results, key=lambda x: x[1]):
#         if r[0] not in seen:
#             unique_results.append(r)
#             seen.add(r[0])
#         if len(unique_results) >= n_results:
#             break

#     return {
#         "query": query or "(image)",
#         "results": [{"id": r[0], "distance": r[1], "metadata": r[2]} for r in unique_results]
#     }

def search(
    query: Optional[str] = None,
    image: Optional[Union[str, Image.Image]] = None,
    n_results: int = 5,
    where: Optional[Dict] = None,
    text_weight: float = 0.5
) -> Dict:
    
    if not query and not image:
        raise ValueError("You must provide at least a text query or an image.")

    where_conditions = [{"active": {"$eq": True}}]
    
    if where:
        where_conditions.append(where)

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
        
        distances = result["distances"][0]
        logger.info(f"Image-only query distances: {distances}")
        
        return {
            "query": "(image)",
            "results": result
        }

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

    where_conditions.append({"modality": {"$eq": "multimodal"}})  
    where_clause = {"$and": where_conditions}
    
    query = translate_to_english(query)
    if not query.strip():
        raise ValueError("Query text is empty after translation.")

    vec_text = embed_text(query)
    image = Image.open(image).convert("RGB") if isinstance(image, str) else image
    vec_image = embed_image(image)
    
    combined_embedding = (text_weight * vec_text + (1 - text_weight) * vec_image)
    combined_embedding = normalize(combined_embedding)

    result = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=n_results,
        where=where_clause
    )
    
    distances = result["distances"][0]
    logger.info(f"Multimodal query distances: {distances}")
    
    return {
        "query": query,
        "results": result
    }