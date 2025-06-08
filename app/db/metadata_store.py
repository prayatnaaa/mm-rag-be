import json, os

DB_FILE = "storage/db.json"

def load_db():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE) as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=2)

def save_source(source_id, url, title, chunk_ids):
    db = load_db()
    db[source_id] = {
        "url": url, "title": title,
        "chunks": chunk_ids, "active": True
    }
    save_db(db)

def list_sources():
    return load_db()
