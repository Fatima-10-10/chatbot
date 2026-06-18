"""
Prompt templates.

Each function returns a ChatPromptTemplate -- a reusable structure with
placeholders filled in at call time, rather than a hardcoded string.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_basic_explainer_prompt() -> ChatPromptTemplate:
    """A simple system + human prompt for short, plain-language explanations."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a friendly teacher who explains things simply, with one short example."),
        ("human", "Explain {topic} in 3 sentences."),
    ])

def get_rag_prompt() -> ChatPromptTemplate:
    """Prompt for answering a question using retrieved context, with history support."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using ONLY the context below. "
         "If the answer isn't in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])