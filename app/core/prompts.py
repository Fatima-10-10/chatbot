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
    """Prompt for RAG with graceful small talk handling."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful document assistant. You have access to context retrieved from uploaded documents.\n\n"
         "Rules:\n"
         "1. If the question is small talk (greetings, how are you, who are you, etc.), respond naturally and briefly — do NOT use the context.\n"
         "2. If the question is about the documents, answer using ONLY the context below. If the answer isn't there, say you don't know.\n"
         "3. ALWAYS respond with this exact JSON format and nothing else:\n"
         '{{"answer": "your answer here", "confidence": "high or medium or low"}}\n\n'
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])