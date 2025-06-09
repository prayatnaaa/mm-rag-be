import faiss
import numpy as np
import os
import pickle

dim = 512
index = faiss.IndexFlatL2(dim)
embedding_map = {}

INDEX_PATH = "faiss_index.index"
EMBED_MAP_PATH = "embedding_map.pkl"

def save_faiss_index():
    faiss.write_index(index, INDEX_PATH)
    with open(EMBED_MAP_PATH, "wb") as f:
        pickle.dump(embedding_map, f)

def load_faiss_index():
    global index, embedding_map
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    if os.path.exists(EMBED_MAP_PATH):
        with open(EMBED_MAP_PATH, "rb") as f:
            embedding_map = pickle.load(f)

def add_embedding(vec, metadata=None):
    eid = f"emb_{len(embedding_map)}"
    arr = np.array([vec]).astype(np.float32)
    index.add(arr)
    embedding_map[eid] = {"vector": vec, "metadata": metadata}
    return eid

def search_similar_chunks(query_vec, top_k=5, allowed_source_ids=None):
    if len(embedding_map) == 0:
        print("⚠️ embedding_map kosong")
        return []

    D, I = index.search(np.array([query_vec]).astype(np.float32), top_k * 5)
    results = []
    keys = list(embedding_map.keys())

    for idx in I[0]:
        if idx < 0 or idx >= len(keys):
            continue

        meta = embedding_map[keys[idx]]["metadata"]
        if allowed_source_ids is None or meta.get("source") in allowed_source_ids:
            results.append(meta)

        if len(results) >= top_k:
            break

    return results
