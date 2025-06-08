import json
import os
from datetime import datetime

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
        "created_at": datetime.utcnow().isoformat()
    }
    _save_metadata(data)

def list_sources():
    return _load_metadata()

def delete_source(source_id):
    data = _load_metadata()
    if source_id in data:
        del data[source_id]
        _save_metadata(data)
        return True
    return False

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
