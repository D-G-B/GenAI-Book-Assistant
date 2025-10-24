"""
FastAPI router for document management endpoints with full file upload support.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db, LoreDocument
from app.schemas.documents import DocumentCreate, DocumentResponse
from typing import Optional
import tempfile
import os
import shutil
from pathlib import Path

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/list", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    """Return a list of all uploaded documents."""
    docs = db.query(LoreDocument).all()
    return docs


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(document: DocumentCreate, db: Session = Depends(get_db)):
    """Upload a new document for the RAG assistant (JSON format - for API use)."""
    new_doc = LoreDocument(**document.model_dump())

    try:
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return new_doc
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create document: {str(e)}")


@router.post("/upload-file", response_model=DocumentResponse)
async def upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a document file (supports PDF, TXT, MD, DOCX, etc.)."""

    # Use filename as title if not provided
    if not title:
        title = file.filename.rsplit('.', 1)[0]

    file_extension = file.filename.split('.')[-1].lower()

    try:
        # Read file content
        file_content = await file.read()

        # Handle different file types
        content = None

        if file_extension in ['txt', 'md', 'markdown']:
            # Text files - decode directly
            content = file_content.decode('utf-8')

        elif file_extension == 'pdf':
            # PDF files - save temporarily and extract text
            print(f"📄 Processing PDF: {file.filename}")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            try:
                # Use PyPDF to extract text
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(temp_path)
                pages = loader.load()

                # Combine all pages
                content = "\n\n".join([page.page_content for page in pages])

                if not content.strip():
                    content = "[PDF processed but no text content extracted]"
                    print(f"⚠️ Warning: No text extracted from PDF {file.filename}")
                else:
                    print(f"✅ Extracted {len(content)} characters from PDF")

            except Exception as e:
                print(f"❌ Error extracting PDF content: {e}")
                content = f"[PDF file: {file.filename} - extraction failed]"
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        elif file_extension in ['docx', 'doc']:
            # Word documents - save temporarily and extract
            print(f"📄 Processing Word document: {file.filename}")

            with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{file_extension}', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(temp_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])

                if not content.strip():
                    content = "[Word document processed but no text content extracted]"
                else:
                    print(f"✅ Extracted {len(content)} characters from Word document")

            except Exception as e:
                print(f"❌ Error extracting Word document: {e}")
                content = f"[Word document: {file.filename} - extraction failed]"
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            # Unknown file type - try to decode as text
            try:
                content = file_content.decode('utf-8')
            except:
                content = f"[Unsupported file type: {file.filename}]"

        # Create document record
        new_doc = LoreDocument(
            title=title,
            filename=file.filename,
            content=content,
            source_type=file_extension
        )

        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        print(f"✅ Document '{title}' uploaded successfully")
        return new_doc

    except Exception as e:
        db.rollback()
        print(f"❌ Failed to upload file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to upload file: {str(e)}")


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


@router.delete("/all")
async def delete_all_documents(db: Session = Depends(get_db)):
    """Delete all documents and reset vector store."""
    try:
        # Delete all documents from database
        db.query(LoreDocument).delete()
        db.commit()

        # Reset vector store
        from app.services.enhanced_rag_service import enhanced_rag_service
        enhanced_rag_service.vector_store = None
        enhanced_rag_service.documents = []
        enhanced_rag_service.processed_documents = {}

        # Delete saved vector store
        vector_store_path = Path("./faiss_index")
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)

        return {"message": "All documents deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to delete documents: {str(e)}")