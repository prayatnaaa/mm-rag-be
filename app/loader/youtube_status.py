from typing import Dict
from threading import Lock

_status_map: Dict[str, str] = {}
_lock = Lock()

def mark_queued(url: str):
    with _lock:
        _status_map[url] = "queued"

def mark_done(url: str):
    with _lock:
        _status_map[url] = "done"

def get_status(url: str) -> str:
    with _lock:
        return _status_map.get(url, "not_found")
