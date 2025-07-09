from datetime import datetime
from typing import Optional
from collections import defaultdict

from app.retriever.chromadb_index import collection


def save_source(source_id: str, path: str, title: str, embedding_ids: list[str]):
    print(f"[CHROMADB] Saved source: {source_id} with {len(embedding_ids)} embeddings")


def list_sources():
    results = collection.get(include=["metadatas"])
    sources = {}

    for meta in results["metadatas"]:
        sid = meta.get("source_id", "unknown")
        title = meta.get("title", "No Title")
        isActive = meta.get("active", False)
        modality = meta.get("modality", "unknown")

        # Inisialisasi struktur awal per source_id
        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "title": title,
                "active": isActive,
                "items": []
            }

        # Tambahkan metadata ke items
        sources[sid]["items"].append({
            "text": meta.get("text"),
            "source_id": sid,
            "title": title,
            "source": meta.get("youtube_url"),
            "image_url": meta.get("image_url"),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "created_at": meta.get("created_at", ""),
            "active": meta.get("active"),
            "modality": modality,
        })

    # Ubah dict menjadi list
    return list(sources.values())

def get_active_sources():
    results = collection.get(where={"active": True}, include=["metadatas"])
    
    return results["metadatas"] if results["ids"] else []



def set_active_status(source_id: str, is_active: bool):
    results = collection.get(where={"source_id": source_id}, include=["metadatas"])
    if not results["ids"]:
        return False

    for i, cid in enumerate(results["ids"]):
        metadata = results["metadatas"][i]
        metadata["active"] = is_active
        collection.update(ids=[cid], metadatas=[metadata])
    return True


def delete_source(source_id):
    try:
        result = collection.get(where={"source_id": source_id}, include=[])
        ids_to_delete = result["ids"]
        if not ids_to_delete:
            return False

        collection.delete(ids=ids_to_delete)
        print(f"🗑️ Deleted source and embeddings: {source_id}")
        return True
    except Exception as e:
        print(f"Failed to delete source: {e}")
        return False



def delete_all_sources():
    try:
        all_ids = collection.get(include=[])["ids"]
        if not all_ids:
            return False
        collection.delete(ids=all_ids)
        print("Deleted all sources and cleared ChromaDB.")
        return {"message": "All sources deleted successfully."}
    except Exception as e:
        print(f"Failed to delete all: {e}")
        return False



def get_chunk_counts_per_source(source_ids: Optional[list] = None):
    if source_ids:
        all_data = collection.get(where={"source_id": {"$in": source_ids}}, include=[])
    else:
        all_data = collection.get(include=[])

    counts = {}
    for sid in [meta["source_id"] for meta in all_data["metadatas"]]:
        counts[sid] = counts.get(sid, 0) + 1

    return counts