"""
FastAPI router for chat/question-answering endpoints.
Supports simplified spoiler filtering with optional reference material.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ServiceStatus

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db),
    document_id: Optional[int] = Query(None, description="Filter search to specific document"),
    max_chapter: Optional[int] = Query(None, description="Spoiler protection: only search up to this chapter (None = full book)"),
    include_reference: bool = Query(False, description="Include reference material (glossary, appendix) when spoiler filter is active")
):
    """
    Ask a question about the uploaded documents.

    Spoiler Protection:
    - max_chapter=None (default): Search entire book, no spoiler filtering
    - max_chapter=15: Only search chapters 1-15
    - include_reference=True: Also search appendices/glossary when spoiler filter is on
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await enhanced_rag_service.ask_question(
            request.question,
            document_id=document_id,
            max_chapter=max_chapter,
            include_reference=include_reference
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@router.get("/status", response_model=ServiceStatus)
async def get_status():
    """Get the current status of the RAG service."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    return ServiceStatus(**enhanced_rag_service.get_status())