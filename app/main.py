from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

    print("Checking handbook ingestion...")
    from app.ingestion.vectorstore import get_vectorstore, upsert_documents, create_index_if_missing
    from app.ingestion.loader import load_and_split

    create_index_if_missing()
    vs = get_vectorstore()
    existing = vs.similarity_search(
        "aurora robotics",
        k=1,
        filter={"source_file": {"$eq": "company_handbook.txt"}}
    )
    if not existing:
        print("Handbook not found in index, ingesting...")
        chunks = load_and_split("data/company_handbook.txt")
        upsert_documents(chunks, "company_handbook.txt")
        print(f"Ingested {len(chunks)} handbook chunks.")
    else:
        print("Handbook already in index, skipping.")

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

# Serve frontend as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def root():
    return FileResponse("frontend/index.html")