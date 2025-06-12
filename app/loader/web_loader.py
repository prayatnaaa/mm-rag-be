import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, urljoin

from app.retriever.embed_clip import embed_text_only, embed_image_only, truncate_clip_text
from app.retriever.faiss_index import add_embedding, save_faiss_index
from app.db.metadata_store import save_source

def load_web_data(url: str, source_id: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.content, "html.parser")
    embedding_ids = []

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    combined_text = "\n".join(paragraphs)
    chunks = truncate_clip_text(combined_text)

    for chunk in chunks:
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    img_tags = soup.find_all("img")
    for idx, img in enumerate(img_tags):
        src = img.get("src")
        if not src:
            continue
        img_url = urljoin(url, src)
        
        if not img_url.startswith("https://"):
            print(f"⚠️ Skip non-https image: {img_url}")
            continue
        try:
            vec = embed_image_only(img_url)
            print(f"saving image url: {img_url}")
            eid = add_embedding(vec, metadata={"source": source_id, "image_url": img_url, "image_idx": idx})
            embedding_ids.append(eid)
        except Exception as e:
            print(f"⚠️ Failed to embed image {src}: {e}")
            continue

    hostname = urlparse(url).hostname or "website"
    save_source(source_id, url, f"Web: {hostname}", embedding_ids)
    save_faiss_index()
    return len(embedding_ids)
