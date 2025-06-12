import os
from PyPDF2 import PdfReader
from app.retriever.embed_clip import embed_text_only, truncate_clip_text, embed_image_only
from app.retriever.faiss_index import add_embedding, save_faiss_index
from app.db.metadata_store import save_source
from app.utils.chunker import chunk_text 
from app.utils.minio_client import get_file_stream_from_minio
# from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "images")

def load_pdf_data(file_path: str, source_id: str):
    reader = PdfReader(file_path)
    all_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text.strip())

    combined_text = "\n".join(all_text)
    chunks = truncate_clip_text(combined_text)

    embedding_ids = []
    for chunk in chunks:
        vec = embed_text_only(chunk)
        eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
        embedding_ids.append(eid)

    save_source(source_id, file_path, f"PDF: {os.path.basename(file_path)}", embedding_ids)
    save_faiss_index() 
    return len(embedding_ids)

def load_pdf_data_from_minio(object_name: str, source_id: str):
    stream = get_file_stream_from_minio(object_name)
    reader = PdfReader(stream)
    embedding_ids = []

    for page in reader.pages:
        text = page.extract_text() or ""
        for chunk in truncate_clip_text(text):
            vec = embed_text_only(chunk)
            eid = add_embedding(vec, metadata={"source": source_id, "text": chunk})
            embedding_ids.append(eid)

    save_source(source_id, f"s3://{MINIO_BUCKET}/{object_name}", f"PDF: {object_name}", embedding_ids)
    return len(embedding_ids)


# def load_pdf_data_from_minio(object_name: str, source_id: str):
#     stream = get_file_stream_from_minio(object_name)
#     reader = PdfReader(stream)
#     stream.seek(0)  # Reset stream position for pdf2image
#     images = convert_from_bytes(stream.read())
#     stream.seek(0)  # Reset again if needed later

#     embedding_ids = []

#     for i, page in enumerate(reader.pages):
#         text = page.extract_text() or ""

#         # ➤ Embed text chunks
#         for chunk in truncate_clip_text(text):
#             vec = embed_text_only(chunk)
#             eid = add_embedding(vec, metadata={"source": source_id, "text": chunk, "page": i})
#             embedding_ids.append(eid)

#         # ➤ Embed rendered image of the page
#         if i < len(images):  # just to be safe
#             img = images[i].convert("RGB")
#             vec = embed_image_only(img)
#             eid = add_embedding(vec, metadata={"source": source_id, "image_page": i})
#             embedding_ids.append(eid)

#     save_source(source_id, f"s3://{MINIO_BUCKET}/{object_name}", f"PDF: {object_name}", embedding_ids)
#     return len(embedding_ids)