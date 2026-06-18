"""
LLM client setup.

Centralizing model construction here means temperature, model name, etc. are
controlled from one place (app/config.py), and if you ever swap providers
(Groq -> OpenAI -> local), only this file changes.
"""

from langchain_groq import ChatGroq

from app.config import settings


def get_llm(temperature: float | None = None) -> ChatGroq:
    """
    Returns a configured Groq chat model.

    temperature can be overridden per-call (e.g. lower for extraction tasks,
    higher for creative ones) without touching global settings.
    """
    return ChatGroq(
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        api_key=settings.groq_api_key,
    )