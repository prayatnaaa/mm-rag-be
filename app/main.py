from fastapi import FastAPI
from app.api import router  
from app.retriever.faiss_index import load_faiss_index

load_faiss_index()
print("✅ FAISS dan embedding_map dimuat")

app = FastAPI()
app.include_router(router)
