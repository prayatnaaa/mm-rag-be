from PIL import Image

def save_frame(img, path):
    Image.fromarray(img).save(path)
