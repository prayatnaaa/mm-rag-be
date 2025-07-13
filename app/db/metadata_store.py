from typing import Optional
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

        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "title": title,
                "active": isActive,
                "length": 0
            }

        sources[sid]["length"] += 1

    return list(sources.values())


def get_active_sources():
    results = collection.get(where={"active": True}, include=["metadatas"])
    sources = {}

    for meta in results.get("metadatas", []):
        sid = meta.get("source_id", "unknown")
        title = meta.get("title", "No Title")
        source = meta.get("source", "unknown")
        active = meta.get("active", "false")
        url = meta.get("youtube_url", "none")

        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "title": title,
                "source": source,
                "url": url,
                "active": active
            }

    return list(sources.values())

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