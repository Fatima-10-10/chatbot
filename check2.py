from app.ingestion.vectorstore import get_vectorstore

vs = get_vectorstore()
results = vs.similarity_search(
    "aurora robotics warehouse robots",
    k=5,
    filter={"source_file": {"$eq": "company_handbook.txt"}}
)
print("Handbook chunks found:", len(results))
for r in results:
    print(r.page_content[:150])
    print()