from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_chat, routes_ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    from app.ingestion.embeddings import get_embeddings
    get_embeddings()
    print("Loading reranker model...")
    from app.rag.reranker import get_reranker
    get_reranker()
    print("All models ready.")
    yield


app = FastAPI(title="LangChain Learning Project", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_chat.router)
app.include_router(routes_ingest.router)


@app.get("/")
def root():
    return {"status": "ok"}