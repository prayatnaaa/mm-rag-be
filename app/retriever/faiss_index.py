import faiss, numpy as np

dim = 512
index = faiss.IndexFlatL2(dim)
embedding_map = {} 

def add_embedding(vec, metadata=None):
    eid = f"emb_{len(embedding_map)}"
    arr = np.array([vec]).astype(np.float32)
    index.add(arr)
    embedding_map[eid] = {"vector": vec, "metadata": metadata}
    return eid

def search_similar_chunks(query_vec, top_k=5):
    if len(embedding_map) == 0:
        return []
    D, I = index.search(np.array([query_vec]).astype(np.float32), top_k)
    results = []
    keys = list(embedding_map.keys())
    for idx in I[0]:
        if idx < 0: continue
        meta = embedding_map[keys[idx]]["metadata"]
        results.append(meta)
    return results
