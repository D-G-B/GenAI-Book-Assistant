"""
Pydantic schemas for Chat and Conversational APIs.
Includes strict typing for simple Q&A and extended types for conversational memory.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Basic Request/Response Models ---

class ChatRequest(BaseModel):
    question: str
    max_chunks: Optional[int] = 3

class ChatSource(BaseModel):
    """Represents a single chunk of text used to answer a question."""
    document_title: str
    chunk_index: int
    similarity_score: float
    # Essential for UI Citations (e.g. "Chapter 5")
    chapter_title: Optional[str] = None
    chapter_number: Optional[int] = None

class ChatResponse(BaseModel):
    """Standard response for Simple Q&A."""
    answer: str
    sources: List[ChatSource]
    confidence: float
    chunks_used: int
    error: Optional[str] = None
    # Essential for UI Spoiler Banner
    spoiler_filter_active: bool = False
    max_chapter: Optional[int] = None

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

# --- Database/History Models (Restored) ---

class ChatHistory(BaseModel):
    """Schema for reading chat history from the database."""
    id: int
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    model_used: str
    created_at: datetime

    class Config:
        from_attributes = True

class ServiceStatus(BaseModel):
    """Schema for the system status endpoint."""
    documents_loaded: int
    total_chunks: int
    embedding_model: str
    vector_database: str
    status: str