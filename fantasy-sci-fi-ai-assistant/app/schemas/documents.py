from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class DocumentBase(BaseModel):
    title: str
    filename: str
    source_type: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None

class DocumentCreate(DocumentBase):
    content: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: int
    content: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentChunkBase(BaseModel):
    chunk_text: str
    chunk_index: int
    chunk_metadata: Optional[Dict[str, Any]] = None

class DocumentChunk(DocumentChunkBase):
    id: int
    document_id: int

    class Config:
        from_attributes = True
