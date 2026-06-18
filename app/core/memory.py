"""
Conversation memory.

Stores chat history per session_id, in process memory. In a real production
service this would be backed by Redis or a database (in-memory dicts disappear
on restart and don't work across multiple server instances) -- but the
*interface* below is exactly what you'd keep even after swapping the backing
store, which is the point of isolating it in its own module.
"""

from langchain_community.chat_message_histories import ChatMessageHistory

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Returns the history for a session, creating a new empty one if needed."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]