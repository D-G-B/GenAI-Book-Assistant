from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

class LoreDocumentBase(BaseModel):
    title: str
    filename: str
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


