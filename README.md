# LangChain Learning Project

A FastAPI service demonstrating core LangChain concepts — Prompts, Models, Output
Parsers, Memory, Documents, Embeddings, Vector DBs, Retrievers, Chains, Runnables,
and RAG — built as a small, modular RAG chatbot rather than disconnected demos.

## Architecture

- `app/core/` — LLM client, prompt templates, output parsers, conversation memory
- `app/ingestion/` — document loading/splitting, embeddings, Pinecone vector store
- `app/rag/` — retriever wrapper, full RAG chain (retrieval + prompt + model + memory)
- `app/api/` — FastAPI routes and request/response schemas
- `app/main.py` — FastAPI app entry point

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
cp .env.example .env        # then fill in your real keys
```

## Running

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Endpoints

- `POST /ingest` — load a text file, chunk it, embed it, and upsert into Pinecone
- `POST /chat` — ask a question; answers are grounded in ingested documents and
  remembered per `session_id`