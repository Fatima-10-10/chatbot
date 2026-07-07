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
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful document assistant.\n\n"
         "Rules:\n"
         "1. For conversational questions (greetings, small talk, questions about "
         "what was said in THIS conversation): answer from the conversation history below.\n"
         "2. For document questions: answer using ONLY the context below. "
         "If the answer isn't in the context, say you don't know.\n"
         "3. CRITICAL: Your ENTIRE response must be ONLY a JSON object. "
         "No text before or after. No explanation. Just raw JSON.\n"
         "4. If the user writes in another language, respond in that language inside the JSON.\n"
         "5. JSON format: "
         '{{\"answer\": \"your answer here\", \"confidence\": \"high or medium or low\"}}\n\n'
         "Context (from documents):\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

def get_rephrase_prompt() -> ChatPromptTemplate:
    """
    Rewrites the user's latest message into a standalone question
    using conversation history, so the retriever doesn't need context to understand it.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the user's latest message, "
         "rewrite the latest message into a standalone question that makes sense "
         "without the conversation history. "
         "If it's small talk or already standalone, return it as-is. "
         "Return ONLY the rewritten question, nothing else."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

def get_query_expansion_prompt() -> ChatPromptTemplate:
    """
    Generates multiple versions of a question to improve retrieval coverage.
    Different phrasings catch different relevant chunks in the vector DB.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         "Generate 3 different versions of the given question that mean the same thing "
         "but use different words and phrasing. This helps retrieve more relevant documents.\n"
         "Return ONLY a JSON array of 3 strings, nothing else.\n"
         'Example: ["version 1", "version 2", "version 3"]'),
        ("human", "{question}"),
    ])