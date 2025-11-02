"""
FastAPI router for document management endpoints - REFACTORED
Uses DocumentManager for all operations to ensure synchronization.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.documents import DocumentCreate, DocumentResponse
from typing import Optional, List
import tempfile

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/list")
async def list_documents(
    include_deleted: bool = Query(False, description="Include soft-deleted documents"),
    db: Session = Depends(get_db)
):
    """
    Return a list of all uploaded documents with their status.

    Args:
        include_deleted: If True, includes soft-deleted documents with status field
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    documents = enhanced_rag_service.document_manager.list_all_documents(
        db,
        include_deleted=include_deleted
    )

    return documents


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(document: DocumentCreate, db: Session = Depends(get_db)):
    """Upload a new document for the RAG assistant (JSON format - for API use)."""
    from app.database import LoreDocument

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
    """
    Upload a document file (supports PDF, TXT, MD, DOCX, etc.).
    Automatically processes the document after upload.
    """
    from app.database import LoreDocument
    from app.services.enhanced_rag_service import enhanced_rag_service

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
                import os
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
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            # Unknown file type - try to decode as text
            try:
                content = file_content.decode('utf-8')
            except:
                content = f"[Unsupported file type: {file.filename}]"

        # Create document record in database
        new_doc = LoreDocument(
            title=title,
            filename=file.filename,
            content=content,
            source_type=file_extension
        )

        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        print(f"✅ Document '{title}' saved to database (ID: {new_doc.id})")

        # Process document through document manager
        success = await enhanced_rag_service.document_manager.add_document(db, new_doc.id)

        if not success:
            # Document was added to DB but processing failed
            print(f"⚠️ Document saved but processing failed")
            raise HTTPException(
                status_code=400,
                detail="Document uploaded but processing failed"
            )

        return new_doc

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to upload file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to upload file: {str(e)}")


@router.delete("/all")
async def delete_all_documents(db: Session = Depends(get_db)):
    """
    Delete all documents.
    Clears database, vector store, and manifest.
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    try:
        success = enhanced_rag_service.document_manager.delete_all_documents(db)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete all documents")

        return {"message": "All documents deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to delete documents: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document (soft delete).
    Removes from database and marks as deleted in vector store.
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    success = enhanced_rag_service.document_manager.delete_document(db, document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found or deletion failed")

    return {"message": f"Document {document_id} successfully deleted"}


@router.post("/{document_id}/process")
async def process_document(document_id: int, db: Session = Depends(get_db)):
    """
    Process an uploaded document for RAG (chunking and embedding).
    Normally called automatically after upload.
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    success = await enhanced_rag_service.document_manager.add_document(db, document_id)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to process document")

    return {"message": "Document processed successfully"}


@router.post("/rebuild-index")
async def rebuild_index(db: Session = Depends(get_db)):
    """
    Rebuild the vector store index from scratch.

    This operation:
    - Physically removes soft-deleted documents from the index
    - Optimizes the index for better performance
    - Reprocesses all active documents

    Use this periodically or when deleted documents exceed 20% of the index.
    """
    from app.services.enhanced_rag_service import enhanced_rag_service

    try:
        print("🔄 Starting index rebuild (this may take a while)...")

        success = await enhanced_rag_service.document_manager.rebuild_index(db)

        if not success:
            raise HTTPException(status_code=400, detail="Index rebuild failed")

        stats = enhanced_rag_service.document_manager.get_stats()

        return {
            "message": "Index rebuilt successfully",
            "stats": stats
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/{document_id}/status")
async def get_document_status(document_id: int, db: Session = Depends(get_db)):
    """Get the status of a document across all systems."""
    from app.services.enhanced_rag_service import enhanced_rag_service
    from app.database import LoreDocument

    # Check if document exists in database
    db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    status = enhanced_rag_service.document_manager.get_document_status(document_id)

    return {
        "id": document_id,
        "title": db_doc.title,
        "filename": db_doc.filename,
        "in_database": True,
        "processed": status['processed'],
        "soft_deleted": status['soft_deleted'],
        "metadata": status['metadata']
    }


@router.get("/stats/overview")
async def get_stats(db: Session = Depends(get_db)):
    """Get overall statistics about document processing."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    stats = enhanced_rag_service.document_manager.get_stats()

    return stats