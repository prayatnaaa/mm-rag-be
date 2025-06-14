from fastapi import FastAPI
from app.api import router  
from app.retriever.faiss_index import load_faiss_index
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_faiss_index()
    print("✅ Loading FAISS index...")
    yield
    print("👋 Shutting down app...")

app = FastAPI(lifespan=lifespan)
app.include_router(router)
