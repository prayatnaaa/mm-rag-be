import faiss
import numpy as np

dim = 512
index = faiss.IndexFlatL2(dim)

# Simpan metadata terkait embedding_id
embedding_map = {}  # eid: {"vector": vec, "metadata": {...}}

def add_embedding(vec, metadata=None):
    eid = f"emb_{len(embedding_map)}"
    index.add(np.array([vec]).astype(np.float32))
    embedding_map[eid] = {"vector": vec, "metadata": metadata}
    return eid

def search_similar_chunks(query_embedding, top_k=5):
    if len(embedding_map) == 0:
        return []

    vec = np.array([query_embedding]).astype(np.float32)
    D, I = index.search(vec, top_k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        eid = list(embedding_map.keys())[idx]
        results.append(embedding_map[eid]["metadata"])
    return results
