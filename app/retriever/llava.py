# from transformers import AutoProcessor, AutoModelForVision2Seq
# from typing import Optional
# from PIL import Image
# import torch

# model_id = "llava-hf/llava-1.5-7b-hf"

# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForVision2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch.float32,  # float32 untuk CPU
#     device_map={"": "cpu"}      # paksa seluruh model ke CPU
# )

# def llava_answer(question: str, image: Optional[Image.Image]):
#     if image is None:
#         raise ValueError("Gambar wajib disediakan untuk LLaVA")

#     prompt = f"<image>\nUSER: {question}\nASSISTANT:"
    
#     inputs = processor(
#         text=prompt,
#         images=image,
#         return_tensors="pt"
#     ).to(model.device, torch.float16)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=512
#     )

#     answer = processor.batch_decode(output, skip_special_tokens=True)[0]
#     return answer
