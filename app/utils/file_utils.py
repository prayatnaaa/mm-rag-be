from PIL import Image
import os
from datetime import datetime
import cv2

def save_frame(img, path):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(path, "JPEG")

def generate_source_id(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]  
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    source_id = f"{base_name}_{timestamp}"
    return source_id
