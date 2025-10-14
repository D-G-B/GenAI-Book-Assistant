"""
API routes for conversational chat with memory.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from uuid import uuid4

from app.database import get_db
from app.schemas.chat import ChatRequest
from app.services.enhanced_rag_service import enhanced_rag_service

router = APIRouter(prefix="/conversation", tags=["Conversational Chat"])


@router.post("/ask")
async def ask_conversational(
    request: ChatRequest,
    session_id: Optional[str] = Query(None, description="Conversation session ID"),
    user_id: Optional[str] = Query(None, description="User identifier"),
    db: Session = Depends(get_db)
):
    """
    Ask a question with conversational memory.
    Follow-up questions will use context from previous messages in the same session.
    """
    if not enhanced_rag_service.context_aware_rag:
        raise HTTPException(
            status_code=400,
            detail="Conversational features not available. Check LLM configuration."
        )

    # Generate session ID if not provided
    if not session_id:
        session_id = f"session_{uuid4()}"

    result = await enhanced_rag_service.context_aware_rag.ask_with_context(
        question=request.question,
        session_id=session_id,
        user_id=user_id
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history for a specific session."""
    if not enhanced_rag_service.context_aware_rag:
        raise HTTPException(status_code=400, detail="Conversational features not available")

    history = enhanced_rag_service.context_aware_rag.get_conversation_history(session_id)

    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    if not enhanced_rag_service.context_aware_rag:
        raise HTTPException(status_code=400, detail="Conversational features not available")

    success = enhanced_rag_service.context_aware_rag.clear_conversation(session_id)

    if success:
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions")
async def list_sessions():
    """List all active conversation sessions."""
    if not enhanced_rag_service.context_aware_rag:
        raise HTTPException(status_code=400, detail="Conversational features not available")

    sessions = enhanced_rag_service.context_aware_rag.list_active_sessions()

    return {
        "sessions": sessions,
        "total": len(sessions)
    }