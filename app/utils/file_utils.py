from PIL import Image
import os
from datetime import datetime

def save_frame(img, path):
    Image.fromarray(img).save(path)

def generate_source_id(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]  
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    source_id = f"{base_name}_{timestamp}"
    return source_id
