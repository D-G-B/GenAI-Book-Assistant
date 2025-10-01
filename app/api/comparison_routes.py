"""
API routes that let you compare both services side-by-side
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse

# Import BOTH services
from app.services.rag_service import rag_service  # Your original
from app.services.langchain_rag_service import langchain_rag_service  # New

router = APIRouter(prefix="/compare", tags=["Comparison"])


@router.post("/ask-original")
async def ask_original(request: ChatRequest, db: Session = Depends(get_db)):
    """Use your original SimpleRAGService"""
    result = await rag_service.ask_question(request.question, db)
    return {
        "service": "Original SimpleRAG",
        "result": result
    }


@router.post("/ask-langchain")
async def ask_langchain(request: ChatRequest, db: Session = Depends(get_db)):
    """Use new LangChain service"""
    result = await langchain_rag_service.ask_question(request.question, db)
    return {
        "service": "LangChain Enhanced",
        "result": result
    }


@router.post("/ask-both")
async def ask_both(request: ChatRequest, db: Session = Depends(get_db)):
    """Ask the same question to BOTH services and compare!"""
    original_result = await rag_service.ask_question(request.question, db)
    langchain_result = await langchain_rag_service.ask_question(request.question, db)

    return {
        "question": request.question,
        "original_service": {
            "service": "SimpleRAG",
            "answer": original_result.get("answer"),
            "chunks_used": original_result.get("chunks_used"),
            "confidence": original_result.get("confidence")
        },
        "langchain_service": {
            "service": "LangChain",
            "answer": langchain_result.get("answer"),
            "chunks_used": langchain_result.get("chunks_used"),
            "confidence": langchain_result.get("confidence")
        }
    }


@router.get("/status")
async def compare_status():
    """Compare status of both services"""
    return {
        "original": rag_service.get_status(),
        "langchain": langchain_rag_service.get_status(),
        "comparison": {
            "both_ready": (
                rag_service.get_status()["status"] == "ready" and
                langchain_rag_service.get_status()["status"] == "ready"
            )
        }
    }