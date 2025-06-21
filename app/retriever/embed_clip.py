from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import base64
from io import BytesIO
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_TOKENS = 72

def embed_text_image(text, image_path=None):
    if isinstance(image_path, Image.Image):
        image = image_path
    elif image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), (255, 255, 255)) 
    
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        text_embedding = outputs.text_embeds
        image_embedding = outputs.image_embeds
        combined = torch.cat([text_embedding, image_embedding], dim=1)
        return combined.squeeze().numpy()


#TODO: Fix the functionality
def embed_text_only(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    return output.squeeze().numpy()


def embed_image_only(image_src):
    if isinstance(image_src, str):
        if image_src.startswith('data:image'):
            try:
                header, base64_data = image_src.split(',', 1)
                image = Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
            except Exception as e:
                raise ValueError(f"Gagal decode base64 image: {e}")
        elif image_src.startswith('http'):
            try:
                resp = requests.get(image_src, timeout=5)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Gagal ambil image dari URL: {e}")
        else:
            raise ValueError("image_src harus berupa base64 atau URL")
    elif isinstance(image_src, Image.Image):
        image = image_src
    else:
        raise TypeError("image_src harus berupa URL, base64 string, atau PIL.Image")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    return output.squeeze().numpy()



def truncate_clip_text(text: str, max_tokens=MAX_TOKENS, stride = 10) -> str:
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - stride 
    return chunks
