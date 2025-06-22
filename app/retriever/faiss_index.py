import faiss
import numpy as np
import os
import pickle
import json

TEXT_DIM = 512
MM_DIM = 512

text_index = None
mm_index = None
text_map = {}
mm_map = {}
initialized = False

TEXT_INDEX_PATH = "faiss_text.index"
MM_INDEX_PATH = "faiss_mm.index"
TEXT_MAP_PATH = "text_map.pkl"
MM_MAP_PATH = "mm_map.pkl"

def init_indices():
    global text_index, mm_index, initialized
    text_index = faiss.IndexFlatL2(TEXT_DIM)
    mm_index = faiss.IndexFlatL2(MM_DIM)
    initialized = True

def save_faiss_indices():
    faiss.write_index(text_index, TEXT_INDEX_PATH)
    faiss.write_index(mm_index, MM_INDEX_PATH)
    with open(TEXT_MAP_PATH, "wb") as f:
        pickle.dump(text_map, f)
    with open(MM_MAP_PATH, "wb") as f:
        pickle.dump(mm_map, f)

def load_faiss_indices():
    global text_index, mm_index, text_map, mm_map
    if os.path.exists(TEXT_INDEX_PATH):
        text_index = faiss.read_index(TEXT_INDEX_PATH)
    if os.path.exists(MM_INDEX_PATH):
        mm_index = faiss.read_index(MM_INDEX_PATH)
    if os.path.exists(TEXT_MAP_PATH):
        with open(TEXT_MAP_PATH, "rb") as f:
            text_map = pickle.load(f)
    if os.path.exists(MM_MAP_PATH):
        with open(MM_MAP_PATH, "rb") as f:
            mm_map = pickle.load(f)

def add_embedding(vec, metadata=None):
    global initialized
    if not initialized:
        init_indices()

    vec = np.array(vec).astype(np.float32)
    eid = f"emb_{len(text_map) + len(mm_map)}"

    if metadata is not None and "source_id" not in metadata and "video_id" in metadata:
        metadata["source_id"] = f"yt_{metadata['video_id']}"

    if vec.shape[0] == TEXT_DIM:
        text_index.add(np.array([vec]))
        text_map[eid] = {"vector": vec, "metadata": metadata}
    elif vec.shape[0] == MM_DIM:
        mm_index.add(np.array([vec]))
        mm_map[eid] = {"vector": vec, "metadata": metadata}
    else:
        raise ValueError(f"Unsupported vector dimension: {vec.shape[0]}")

    return eid

def search_similar_chunks(query_vec, top_k=5, allowed_source_ids=None):
    print("Searching... TEXT:", text_index.ntotal, " | MM:", mm_index.ntotal)
    
    global initialized
    if not initialized:
        load_faiss_indices()

    query_vec = np.array([query_vec]).astype(np.float32)
    qdim = query_vec.shape[1]
    results = []

    def _search(index, emap, label=""):
        if index.ntotal == 0:
            print(f"⚠️ {label} index is empty.")
            return []
        
        D, I = index.search(query_vec, top_k * 5)
        keys = list(emap.keys())
        matches = []

        for idx_pos, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(keys):
                continue
            
            key = keys[idx]
            meta = emap[key]["metadata"]
            if meta is None:
                continue

            video_id = meta.get("video_id")
            allowed_video_ids = [sid.replace("yt_", "") for sid in allowed_source_ids] if allowed_source_ids else None
            
            if allowed_source_ids is None or video_id in allowed_video_ids:
                meta_copy = dict(meta)
                meta_copy["distance"] = float(D[0][idx_pos])  # L2 distance
                matches.append(meta_copy)
        
        return matches

    if qdim == TEXT_DIM:
        results = _search(text_index, text_map, label="TEXT")
    elif qdim == MM_DIM:
        results = _search(mm_index, mm_map, label="MM")
    else:
        raise ValueError(f"Unsupported query vector dimension: {qdim}")

    results_sorted = sorted(results, key=lambda x: x["distance"])

    if len(results_sorted) < top_k:
        print(f"⚠️ Only {len(results_sorted)} valid results after filtering (requested: {top_k}).")

    print(f"🔍 Top {min(len(results_sorted), top_k)} results (sorted by distance):")
    for r in results_sorted[:top_k]:
        print(f" - {r['title']} | video_id: {r['video_id']} | distance: {r['distance']:.4f}")

    return results_sorted[:top_k]


save_faiss_index = save_faiss_indices
load_faiss_index = load_faiss_indices