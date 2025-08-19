"""
FastAPI router for document management endpoints.

These endpoints will handle:
- Uploading new documents
- Listing existing documents
- Deleting documents by ID

The routes will use:
- `crud.py` for database operations
- `document_service.py` for file loading and chunking
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import crud
from app.lore_schemas import LoreDocumentCreate

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/list")
async def list_documents(db: Session = Depends(get_db)):
    """
    Return a list of all uploaded documents.
    """
    docs = crud.get_all_documents(db)
    return docs


@router.post("/upload")
async def upload_document(document: LoreDocumentCreate, db: Session = Depends(get_db)):
    """
    Upload a new document for the lore assistant.
    Steps (future implementation):
    1. Save file temporarily
    2. Pass file to document_service for processing
    3. Store chunks in vector DB
    4. Store metadata in SQL database
    """
    new_doc = (crud.create_lore_document(db, document))
    return new_doc


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document from the database (and possibly from vector DB).
    """
    deleted_doc = crud.delete_document(db, document_id)
    if not deleted_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return  {"message": f"Document with title '{deleted_doc.title}' successfully deleted."}
