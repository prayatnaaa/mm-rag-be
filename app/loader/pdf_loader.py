import os
import fitz
from PyPDF2 import PdfReader
from io import BytesIO
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_bytes_to_minio
from app.utils.text_utils import chunk_text, clean_metadata
from app.retriever.chromadb_index import embed_text, embed_image, add_embedding_pdf

def load_pdf_data(pdf_path: str, source_id: str):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    object_name = f"pdfs/{source_id}.pdf"
    source_url = upload_bytes_to_minio(
        file_bytes=pdf_bytes,
        object_name=object_name,
        content_type="application/pdf"
    )

    embeddings = []

    reader = PdfReader(pdf_path)

    for page_no, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        chunks = chunk_text([
            {"text": t.strip(), "start": page_no, "end": page_no}
            for t in text.split('\n') if t.strip()
        ])

        for chunk in chunks:
            vec = embed_text(chunk["text"])
            meta = {
                "source_id": source_id or source_url,
                "text": chunk["text"],
                "title": os.path.basename(pdf_path),
                "source": "pdf",
                "source_url": source_url,
                "active": True,
            }
            eid = add_embedding_pdf(vec, clean_metadata(meta | {"modality": "text"}))
            embeddings.append(eid)

    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_mime = f"image/{image_ext}"

            image_object_name = f"{source_id}/page_{page_index}_img_{img_index}.{image_ext}"
            image_url = upload_bytes_to_minio(
                file_bytes=image_bytes,
                object_name=image_object_name,
                content_type=image_mime
            )

            image_vec = embed_image(BytesIO(image_bytes))
            image_meta = {
                "source_id": source_id,
                "image_url": image_url,
                "page": page_index,
                "title": os.path.basename(pdf_path),
                "source": "pdf",
                "source_url": source_url,
                "active": True,
            }
            eid = add_embedding_pdf(image_vec, clean_metadata(image_meta | {"modality": "image"}))
            embeddings.append(eid)

    doc.close()

    save_source(source_id, f"s3://{object_name}", os.path.basename(pdf_path), embeddings)

    return {"status": "ok", "chunks": len(embeddings)}