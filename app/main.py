"""
FastAPI application entry point.
"""

from fastapi import FastAPI

from app.api import routes_chat, routes_ingest

app = FastAPI(title="LangChain Learning Project")

app.include_router(routes_chat.router)
app.include_router(routes_ingest.router)


@app.get("/")
def root():
    return {"status": "ok"}