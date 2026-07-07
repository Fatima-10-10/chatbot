"""
Reranking: after Pinecone returns top-k chunks by vector similarity,
a cross-encoder reranker scores each (query, chunk) pair directly
and reorders them by true relevance.

Why this is better than pure vector similarity:
- Vector similarity compares embeddings independently (query vector vs chunk vector)
- Cross-encoder reads query + chunk TOGETHER, understanding their relationship
- Much more accurate but slower, so we only run it on the small set Pinecone already filtered
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Lightweight cross-encoder that runs well on CPU
# Downloads once (~90MB), cached locally after that
_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Reranks documents by true relevance to the query.
    Returns top_n most relevant docs in order.
    
    Flow: Pinecone returns 10 candidates → reranker scores all 10 
    → return top 5 in correct relevance order
    """
    if not docs:
        return docs

    reranker = get_reranker()

    # Score each (query, chunk) pair
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # Sort by score descending, return top_n
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]