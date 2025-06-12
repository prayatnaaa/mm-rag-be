from fastapi import FastAPI
from app.api import router  
from app.retriever.faiss_index import load_faiss_index
from contextlib import asynccontextmanager

# load_faiss_index()
# print("✅ FAISS dan embedding_map dimuat")
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Loading FAISS index...")
    load_faiss_index()
    yield
    print("👋 Shutting down app...")

app = FastAPI(lifespan=lifespan)
app.include_router(router)
