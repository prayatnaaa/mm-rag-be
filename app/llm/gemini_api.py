import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_answer_from_gemini(contexts, query):
    contexts_text = "\n".join([f"- {ctx['text']} (Image: {ctx['image_url']})" for ctx in contexts if ctx])
    prompt = f"""Gunakan informasi berikut untuk menjawab pertanyaan dengan lengkap:
{contexts_text}

Pertanyaan: {query}"""
    response = model.generate_content(prompt)
    return {
        "answer": response.text,
        "evidence": contexts
    }
