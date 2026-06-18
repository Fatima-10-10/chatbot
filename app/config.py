"""
Centralized application settings.

All configuration is read from environment variables (via .env in development).
This is the ONLY module that should touch os.environ / .env directly --
everywhere else imports `settings` from here instead.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- API keys ---
    groq_api_key: str
    pinecone_api_key: str

    # --- LLM config ---
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # --- Embeddings + Vector DB ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    pinecone_index_name: str = "langchain-learning"

    # --- Document ingestion ---
    chunk_size: int = 300
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()