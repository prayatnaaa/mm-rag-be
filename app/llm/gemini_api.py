import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import requests
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_answer_from_gemini(contexts, query):
    parts = []

    # Tambahkan instruksi dan pertanyaan
    parts.append("Berikut ini adalah potongan transkrip video dan gambar. Jawablah pertanyaan user secara jelas dan ringkas.")
    parts.append(f"[PERTANYAAN USER]\n{query}")

    # Tambahkan konteks gambar dan teks
    for ctx in contexts:
        text = ctx.get("text", "")
        image_url = ctx.get("image_url")

        if image_url:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image_bytes = response.content
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
            except Exception as e:
                print(f"❌ Gagal unduh gambar: {image_url} — {e}")

        # Tambahkan teks transkrip
        if text:
            parts.append(f"[KONTEKS VIDEO]\n{text}")

    # Kirim ke Gemini
    response = model.generate_content(parts)
    return {"answer": response.text, "evidence": contexts}

