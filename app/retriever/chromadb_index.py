import chromadb
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np
from deep_translator import GoogleTranslator
from typing import Optional, Dict, Union

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

def embed_image(image_path: Union[str, Image.Image]) -> np.ndarray:
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

def search(query: Optional[str] = None, image: Optional[Union[str, Image.Image]] = None, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
    if not query and not image:
        raise ValueError("You must provide at least a text query or an image.")

    results = {
        "query": query,
        "text_results": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
        "image_results": {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    }

    use_text = query and query.strip().lower() not in {
        "jelaskan gambar ini", "apa isi gambar ini", "apa yang terjadi di gambar ini", "describe this image"
    }

    if image:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_embedding = embed_image(image)

        results["text_results"] = collection.query(
            query_embeddings=[image_embedding.tolist()],
            n_results=n_results,
            where={"$and": [{"modality": {"$eq": "text"}}, {"active": {"$eq": True}}, where]} if where else {"$and": [{"modality": {"$eq": "text"}}, {"active": {"$eq": True}}]}
        )

        results["image_results"] = collection.query(
            query_embeddings=[image_embedding.tolist()],
            n_results=n_results,
            where={"$and": [{"modality": {"$eq": "image"}}, {"active": {"$eq": True}}, where]} if where else {"$and": [{"modality": {"$eq": "image"}}, {"active": {"$eq": True}}]}
        )

    elif use_text:
        text_embedding = embed_text(query)

        results["text_results"] = collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results,
            where={"$and": [{"modality": {"$eq": "text"}}, {"active": {"$eq": True}}, where]} if where else {"$and": [{"modality": {"$eq": "text"}}, {"active": {"$eq": True}}]}
        )

        results["image_results"] = collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results,
            where={"$and": [{"modality": {"$eq": "image"}}, {"active": {"$eq": True}}, where]} if where else {"$and": [{"modality": {"$eq": "image"}}, {"active": {"$eq": True}}]}
        )

    return results