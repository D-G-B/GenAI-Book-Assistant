"""Tests for per-user ownership of conversation sessions.

ContextAwareRAG.get_conversation_history / clear_conversation / list_active_sessions
must only ever touch sessions owned by the requesting user. A non-owned session is
treated as absent so a caller can't confirm another user's session exists.

The RAG is built via __new__ with just a ConversationMemoryManager attached —
none of the LLM/vector-store machinery is needed for the ownership checks.
"""

from app.services.conversational_memory import (
    ContextAwareRAG,
    ConversationMemoryManager,
)


def make_rag():
    rag = ContextAwareRAG.__new__(ContextAwareRAG)
    rag.memory_manager = ConversationMemoryManager()
    return rag


def test_history_scoped_to_owner():
    rag = make_rag()
    session = rag.memory_manager.get_or_create_session("s1", user_id=1)
    session.add_message("human", "hi")
    session.add_message("assistant", "hello")

    assert len(rag.get_conversation_history("s1", user_id=1)) == 2
    # A different user cannot read it (and can't tell it exists).
    assert rag.get_conversation_history("s1", user_id=2) == []


def test_clear_blocks_foreign_user():
    rag = make_rag()
    rag.memory_manager.get_or_create_session("s1", user_id=1)

    assert rag.clear_conversation("s1", user_id=2) is False
    assert "s1" in rag.memory_manager.sessions  # not removed

    assert rag.clear_conversation("s1", user_id=1) is True
    assert "s1" not in rag.memory_manager.sessions


def test_clear_missing_session_returns_false():
    rag = make_rag()
    assert rag.clear_conversation("nope", user_id=1) is False


def test_list_sessions_scoped_to_user():
    rag = make_rag()
    rag.memory_manager.get_or_create_session("s1", user_id=1)
    rag.memory_manager.get_or_create_session("s2", user_id=2)
    rag.memory_manager.get_or_create_session("s3", user_id=1)

    ids = {s["session_id"] for s in rag.list_active_sessions(user_id=1)}
    assert ids == {"s1", "s3"}
