from minio import Minio
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "images")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def ensure_bucket_exists():
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

def upload_image(local_path, object_name):
    ensure_bucket_exists()
    client.fput_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        file_path=local_path,
        content_type="image/jpeg"
    )
    return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"
