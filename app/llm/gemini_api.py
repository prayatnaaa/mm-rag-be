import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_answer_from_gemini(contexts, query):
    prompt = "Gunakan konteks dari sumber berikut:\n"
    print('hhhhh')
    for ctx in contexts:
        print ("context ", ctx)
        prompt += f"- {ctx.get('text','')} (sumber: {ctx.get('source','')})\n"
    prompt += f"\nPertanyaan: {query}"
    
    res = model.generate_content(prompt)
    return {"answer": res.text, "evidence": contexts}
