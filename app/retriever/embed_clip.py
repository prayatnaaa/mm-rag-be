from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_TOKENS = 72

def embed_text_image(text, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**inputs).pooler_output
    return output.squeeze().numpy()

def embed_text_only(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    return output.squeeze().numpy()

def truncate_clip_text(text: str, max_tokens=MAX_TOKENS, stride = 10) -> str:
    # tokens = tokenizer.tokenize(text)
    # chunks = []
    # for i in range(0, len(tokens), max_tokens):
    #     chunk_tokens = tokens[i:i+max_tokens]
    #     chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
    #     chunks.append(chunk_text)
    # return chunks
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
