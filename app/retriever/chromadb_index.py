import chromadb
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np
from deep_translator import GoogleTranslator
from typing import Optional

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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
        print(f"[Translation Error] {e}")
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

def embed_image(image_path: str | Image.Image) -> np.ndarray:
    if isinstance(image_path, Image.Image):
        image = image_path
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze().numpy()
    return normalize(embedding)

def add_embedding(vec: np.ndarray, metadata: dict) -> str:
    if "source_id" not in metadata and "video_id" in metadata:
        metadata["source_id"] = f"yt_{metadata['video_id']}"

    doc_id = f"{metadata['modality']}_{metadata['video_id']}_{int(metadata['start_time'] * 1000)}"

    collection.add(
        documents=[metadata.get("text", "(image)")],
        embeddings=[vec.tolist()],
        metadatas=[metadata],
        ids=[doc_id],
    )
    return doc_id

# def search(query: str, n_results=5, modality=None):
#     query_embedding = embed_text(query)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=n_results,
#         where={"modality": modality} if modality else None
#     )
#     return results

def search(query: Optional[str] = None, image: Optional[str | Image.Image] = None, n_results: int = 5):

    if not query and not image:
        raise ValueError("You must provide at least a text query or an image.")

    query_embeddings = []

    if query:
        text_embedding = embed_text(query)
        query_embeddings.append(text_embedding)

    if image:
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Invalid image format. Must be path or PIL.Image.")

        image_embedding = embed_image(pil_image)
        query_embeddings.append(image_embedding)

    combined_embedding = normalize(np.mean(query_embeddings, axis=0))

    text_results = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=n_results,
        where={
            "$and": [
                {"modality": {"$eq": "text"}},
                {"active": {"$eq": True}}
            ]
        }
    )

    image_results = collection.query(
        query_embeddings=[combined_embedding.tolist()],
        n_results=n_results,
        where={
            "$and": [
                {"modality": {"$eq": "image"}},
                {"active": {"$eq": True}}
            ]
        }   
    )

    return {
        "query": query,
        "text_results": text_results,
        "image_results": image_results
    }