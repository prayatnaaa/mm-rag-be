import json
import os
from datetime import datetime
import numpy as np

from app.retriever.faiss_index import (
    text_map,
    mm_map,
    save_faiss_index,
    load_faiss_index,
    init_indices,
    text_index,
    mm_index,
)

DB_FILE = "storage/db.json"


def _load_metadata():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_source(source_id, path, title, embedding_ids):
    data = _load_metadata()
    data[source_id] = {
        "path": path,
        "title": title,
        "embedding_ids": embedding_ids,
        "active": True,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_metadata(data)


def list_sources():
    return _load_metadata()


def delete_source(source_id):
    data = _load_metadata()
    if source_id not in data:
        return False

    load_faiss_index()

    embedding_ids = data[source_id].get("embedding_ids", [])

    text_ids = []
    mm_ids = []

    for eid in embedding_ids:
        if eid in text_map:
            index = int(list(text_map.keys()).index(eid))
            text_ids.append(index)
            del text_map[eid]
        elif eid in mm_map:
            index = int(list(mm_map.keys()).index(eid))
            mm_ids.append(index)
            del mm_map[eid]

    if text_index and text_ids:
        text_index.remove_ids(np.array(text_ids, dtype=np.int64))
    if mm_index and mm_ids:
        mm_index.remove_ids(np.array(mm_ids, dtype=np.int64))

    save_faiss_index()

    del data[source_id]
    _save_metadata(data)
    print(f"🗑️ Deleted source and embeddings: {source_id}")
    return True


def delete_all_sources():
    data = _load_metadata()
    if not data:
        return False

    load_faiss_index()

    text_map.clear()
    mm_map.clear()

    if text_index is not None:
        text_index.reset()
    else:
        init_indices()

    if mm_index is not None:
        mm_index.reset()
    else:
        init_indices()

    save_faiss_index()
    _save_metadata({})
    print("🧹 Deleted all sources and cleared FAISS.")
    return True


def set_active_status(source_id, is_active: bool):
    data = _load_metadata()
    if source_id in data:
        data[source_id]["active"] = is_active
        _save_metadata(data)
        return True
    return False


def get_active_sources():
    data = _load_metadata()
    return {sid: meta for sid, meta in data.items() if meta.get("active", True)}