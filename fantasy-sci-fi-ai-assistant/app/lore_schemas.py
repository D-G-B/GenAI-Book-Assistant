from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

class LoreDocumentBase(BaseModel):
    title: str
    filename: str
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LoreDocumentCreate(LoreDocumentBase):
    content: Optional[str] = None

class LoreDocument(LoreDocumentBase):
    id: int
    content: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentChunkBase(BaseModel):
    chunk_text: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None

class DocumentChunk(DocumentChunkBase):
    id: int
    document_id: int

    class Config:
        from_attributes = True
