"""
FastAPI router for document management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, LoreDocument
from app.schemas.documents import DocumentCreate, DocumentResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/list", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    """Return a list of all uploaded documents."""
    docs = db.query(LoreDocument).all()
    return docs


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(document: DocumentCreate, db: Session = Depends(get_db)):
    """Upload a new document for the RAG assistant."""
    new_doc = LoreDocument(**document.model_dump())

    try:
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return new_doc
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create document: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document from the database."""
    doc_to_delete = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

    if not doc_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        db.delete(doc_to_delete)
        db.commit()
        return {"message": f"Document '{doc_to_delete.title}' successfully deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to delete document: {str(e)}")


@router.post("/{document_id}/process")
async def process_document(document_id: int, db: Session = Depends(get_db)):
    """Process uploaded document for RAG (chunking and embedding)."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    success = await enhanced_rag_service.process_document(db, document_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to process document")

    return {"message": "Document processed successfully"}