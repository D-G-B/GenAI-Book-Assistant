"""
FastAPI router for document management endpoints - REFACTORED
Uses DocumentManager for all operations to ensure synchronization.
Now supports EPUB files.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.documents import DocumentCreate, DocumentResponse
from typing import Optional, List
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["Documents"])

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'txt', 'md', 'markdown',  # Text formats
    'pdf',                      # PDF
    'docx', 'doc',              # Word
    'csv', 'json',              # Data formats
    'epub'                      # E-books
}


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


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "formats": list(SUPPORTED_EXTENSIONS),
        "descriptions": {
            "txt": "Plain text files",
            "md": "Markdown files",
            "markdown": "Markdown files",
            "pdf": "PDF documents",
            "docx": "Microsoft Word documents",
            "doc": "Microsoft Word documents (legacy)",
            "csv": "Comma-separated values",
            "json": "JSON files",
            "epub": "E-book files (EPUB format)"
        }
    }


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
    Upload a document file (supports PDF, TXT, MD, DOCX, EPUB, etc.).
    Automatically processes the document after upload.
    """
    from app.database import LoreDocument
    from app.services.enhanced_rag_service import enhanced_rag_service

    # Use filename as title if not provided
    if not title:
        title = file.filename.rsplit('.', 1)[0]

    file_extension = file.filename.split('.')[-1].lower()

    # Validate file extension
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_extension}. "
                   f"Supported formats: {', '.join('.' + ext for ext in SUPPORTED_EXTENSIONS)}"
        )

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
            content = await _extract_pdf_content(file_content, file.filename)

        elif file_extension in ['docx', 'doc']:
            # Word documents - save temporarily and extract
            print(f"📄 Processing Word document: {file.filename}")
            content = await _extract_word_content(file_content, file.filename, file_extension)

        elif file_extension == 'epub':
            # EPUB files - save temporarily and extract
            print(f"📚 Processing EPUB: {file.filename}")
            content = await _extract_epub_content(file_content, file.filename)

        elif file_extension in ['csv', 'json']:
            # Data files - decode as text
            content = file_content.decode('utf-8')

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


async def _extract_pdf_content(file_content: bytes, filename: str) -> str:
    """Extract text content from PDF file."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # Combine all pages
        content = "\n\n".join([page.page_content for page in pages])

        if not content.strip():
            content = "[PDF processed but no text content extracted]"
            print(f"⚠️ Warning: No text extracted from PDF {filename}")
        else:
            print(f"✅ Extracted {len(content)} characters from PDF")

        return content

    except Exception as e:
        print(f"❌ Error extracting PDF content: {e}")
        return f"[PDF file: {filename} - extraction failed]"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _extract_word_content(file_content: bytes, filename: str, extension: str) -> str:
    """Extract text content from Word document."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{extension}', delete=False) as temp_file:
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

        return content

    except Exception as e:
        print(f"❌ Error extracting Word document: {e}")
        return f"[Word document: {filename} - extraction failed]"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _extract_epub_content(file_content: bytes, filename: str) -> str:
    """
    Extract text content from EPUB file.
    Uses ebooklib for chapter-aware extraction with BeautifulSoup for HTML parsing.
    """
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.epub', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        # Try ebooklib first (preferred for chapter awareness)
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(temp_path)

            # Get book metadata
            book_title = book.get_metadata('DC', 'title')
            book_title = book_title[0][0] if book_title else filename

            book_author = book.get_metadata('DC', 'creator')
            book_author = book_author[0][0] if book_author else 'Unknown'

            print(f"📖 EPUB: {book_title} by {book_author}")

            # Extract text from all document items
            chapters = []
            chapter_count = 0

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    html_content = item.get_content()
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Extract text from paragraphs
                    text_parts = []
                    for tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4']:
                        for element in soup.find_all(tag):
                            text = element.get_text(' ', strip=True)
                            if text and len(text) > 5:
                                text_parts.append(text)

                    chapter_text = '\n\n'.join(text_parts)

                    # Only include substantial chapters
                    if len(chapter_text) > 100:
                        chapter_count += 1
                        chapters.append(f"--- Chapter {chapter_count} ---\n\n{chapter_text}")

            content = '\n\n'.join(chapters)

            if not content.strip():
                content = "[EPUB processed but no text content extracted]"
                print(f"⚠️ Warning: No text extracted from EPUB {filename}")
            else:
                print(f"✅ Extracted {len(content)} characters from {chapter_count} chapters")

            return content

        except ImportError:
            # Fallback to UnstructuredEPubLoader
            print("⚠️ ebooklib not available, trying UnstructuredEPubLoader...")
            from langchain_community.document_loaders import UnstructuredEPubLoader

            loader = UnstructuredEPubLoader(temp_path, mode="single")
            docs = loader.load()
            content = "\n\n".join([doc.page_content for doc in docs])

            if not content.strip():
                content = "[EPUB processed but no text content extracted]"
            else:
                print(f"✅ Extracted {len(content)} characters from EPUB (unstructured)")

            return content

    except Exception as e:
        print(f"❌ Error extracting EPUB content: {e}")
        import traceback
        traceback.print_exc()
        return f"[EPUB file: {filename} - extraction failed: {str(e)}]"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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