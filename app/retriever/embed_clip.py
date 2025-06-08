from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
