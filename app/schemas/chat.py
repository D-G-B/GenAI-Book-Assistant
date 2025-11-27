from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    question: str
    max_chunks: Optional[int] = 3

class ChatSource(BaseModel):
    document_title: str
    chunk_index: int
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    confidence: float
    chunks_used: int
    error: Optional[str] = None
    spoiler_filter_active: bool = False
    max_chapter: Optional[int] = None

class ChatHistory(BaseModel):
    id: int
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    model_used: str
    created_at: datetime

    class Config:
        from_attributes = True

class ServiceStatus(BaseModel):
    documents_loaded: int
    total_chunks: int
    embedding_model: str
    vector_database: str
    status: str
