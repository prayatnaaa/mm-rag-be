import chromadb
from chromadb.utils import embedding_functions
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np

# Load CLIP model dan processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

MAX_TOKENS = 72

# Inisialisasi ChromaDB client
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="embeddings", embedding_function=None)

# ✅ Fungsi untuk normalisasi L2 (agar dot product = cosine similarity)
def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# 🔤 Embedding teks dengan normalisasi
def embed_text(text: str, truncate=True) -> np.ndarray:
    if truncate:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
            text = tokenizer.convert_tokens_to_string(tokens)

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).squeeze().numpy()
    return normalize(embedding)

# 🖼️ Embedding gambar dengan normalisasi
def embed_image(image_path: str | Image.Image) -> np.ndarray:
    if isinstance(image_path, Image.Image):
        image = image_path
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze().numpy()
    return normalize(embedding)

# ➕ Menambahkan embedding ke ChromaDB
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

# 🔍 Pencarian berdasarkan teks query
def search(query: str, n_results=5, modality=None):
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where={"modality": modality} if modality else None
    )
    return results