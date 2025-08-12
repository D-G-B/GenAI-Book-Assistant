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

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a new document for the lore assistant.
    Steps (future implementation):
    1. Save file temporarily
    2. Pass file to document_service for processing
    3. Store chunks in vector DB
    4. Store metadata in SQL database
    """
    # Placeholder response until implemented
    return {"filename": file.filename, "status": "upload received"}

@router.get("/list")
async def list_documents(db: Session = Depends(get_db)):
    """
    Return a list of all uploaded documents.
    """
    docs = crud.get_all_documents(db)
    return docs

@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document from the database (and possibly from vector DB).
    """
    # TODO: Implement deletion in crud.py
    raise HTTPException(status_code=501, detail="Not implemented yet")
