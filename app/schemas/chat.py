"""
Pydantic schemas for Chat and Conversational APIs.
Includes simplified spoiler model with reference material toggle.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from app.config import settings

# --- Basic Request/Response Models ---

class ChatRequest(BaseModel):
    # Cap question length to bound per-query token use (whitespace-only is still
    # rejected in the route). max_length yields a 422 with a clear message.
    question: str = Field(..., max_length=settings.MAX_QUESTION_LENGTH)

class ChatSource(BaseModel):
    """Represents a single chunk of text used to answer a question."""
    document_title: str
    chunk_index: int
    # Cosine similarity in [0, 1]. Optional because callers that don't pass
    # through search_with_scores (legacy paths, manually constructed responses)
    # may omit it.
    similarity_score: Optional[float] = None
    chapter_title: Optional[str] = None
    chapter_number: Optional[int] = None
    is_reference: Optional[bool] = None  # True if from glossary/appendix

class ChatResponse(BaseModel):
    """Standard response for Simple Q&A."""
    answer: str
    sources: List[ChatSource]
    # Mean of source similarity scores; None when scores aren't available.
    confidence: Optional[float] = None
    chunks_used: int
    error: Optional[str] = None
    # Spoiler info
    spoiler_filter_active: bool = False
    max_chapter: Optional[int] = None
    include_reference: Optional[bool] = None
    # Telemetry: which provider answered, how many LLM calls this request made.
    llm_provider: Optional[str] = None
    llm_calls: Optional[int] = None

# --- Extended Models for Conversational Mode ---

class ConversationResponse(ChatResponse):
    """
    Extended response for Conversational Mode.
    Inherits all fields from ChatResponse and adds session context.
    """
    session_id: str
    conversation_length: int
    context_used: bool
    filtered_to_document: Optional[int] = None

class ServiceStatus(BaseModel):
    """Schema for the system status endpoint."""
    documents_loaded: int
    total_chunks: int
    embedding_model: str
    vector_database: str
    status: str
    # Cumulative LLM call counters since server startup (in-memory only).
    llm_calls_total: Optional[int] = None
    llm_calls_by_provider: Optional[Dict[str, int]] = None