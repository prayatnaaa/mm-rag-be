from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import base64
import numpy as np
from io import BytesIO
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_TOKENS = 72

def truncate_clip_text(text: str, max_tokens=MAX_TOKENS, stride=10) -> list[str]:
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

def embed_text_only(text: str) -> np.ndarray:
    chunks = truncate_clip_text(text)
    embeddings = []
    for chunk in chunks:
        inputs = processor(text=[chunk], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            embeddings.append(outputs.squeeze().numpy())
    return np.mean(embeddings, axis=0)

def embed_image_only(image_src) -> np.ndarray:
    if isinstance(image_src, str):
        if image_src.startswith('data:image'):
            header, base64_data = image_src.split(',', 1)
            image = Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
        elif image_src.startswith('http'):
            resp = requests.get(image_src, timeout=5)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            raise ValueError("image_src harus berupa base64 atau URL")
    elif isinstance(image_src, Image.Image):
        image = image_src
    else:
        raise TypeError("image_src harus berupa URL, base64 string, atau PIL.Image")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.squeeze().numpy()

def embed_text_image(text: str, image_path=None) -> dict:
    if isinstance(image_path, Image.Image):
        image = image_path
    elif image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), (255, 255, 255))

    text_chunks = truncate_clip_text(text)
    text_embeddings = []
    for chunk in text_chunks:
        inputs = processor(text=[chunk], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            output = model(**inputs)
            text_embeddings.append(output.text_embeds.squeeze().numpy())

    pooled_text_embedding = np.mean(text_embeddings, axis=0)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).squeeze().numpy()

    return {
        "text_embedding": pooled_text_embedding,
        "image_embedding": image_embedding
    }