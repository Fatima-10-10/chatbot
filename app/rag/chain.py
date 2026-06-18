"""
RAG chain: ties together Retriever + Prompt + Model + Memory into one Runnable.

Every piece here -- the retriever, the prompt, the model -- implements the same
Runnable interface (.invoke()), which is exactly what lets them be composed with
`|` regardless of what each one actually does underneath.
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.llm import get_llm
from app.core.memory import get_session_history
from app.core.prompts import get_rag_prompt
from app.rag.retriever import get_retriever


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chatbot() -> RunnableWithMessageHistory:
    """
    Returns a chatbot Runnable: given {"input": "<question>"} and a session_id
    in config, retrieves relevant context, answers grounded in it, and remembers
    the conversation across calls for that session_id.
    """
    retriever = get_retriever(k=3)
    model = get_llm()
    prompt = get_rag_prompt()

    # RunnablePassthrough.assign() keeps the original input dict intact (so 'history'
    # injected later survives) while ADDING a 'context' key computed from retrieval.
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | model
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )