from app.ingestion.vectorstore import create_index_if_missing
from pinecone import Pinecone
from app.config import settings

print("Creating index if missing...")
create_index_if_missing()
print("Done!")

pc = Pinecone(api_key=settings.pinecone_api_key)
indexes = [idx.name for idx in pc.list_indexes()]
print("Existing indexes:", indexes)