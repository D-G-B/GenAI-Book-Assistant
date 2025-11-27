"""
FastAPI router for document management endpoints
Now supports EPUB files.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.documents import DocumentCreate, DocumentResponse
from typing import Optional
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/list")
async def list_documents(
    include_deleted: bool = Query(False, description="Include soft-deleted documents"),
    db: Session = Depends(get_db)
):
    """Return a list of all uploaded documents with their status."""
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
    Upload a document file (PDF, TXT, MD, DOCX, EPUB, etc.).
    Automatically processes the document after upload.
    """
    from app.database import LoreDocument
    from app.services.enhanced_rag_service import enhanced_rag_service

    # Use filename as title if not provided
    if not title:
        title = file.filename.rsplit('.', 1)[0]

    file_extension = file.filename.split('.')[-1].lower()

    # Supported file types
    supported = {'txt', 'md', 'markdown', 'pdf', 'docx', 'doc', 'csv', 'json', 'epub'}
    if file_extension not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_extension}. Supported: {', '.join(supported)}"
        )

    try:
        file_content = await file.read()
        content = None

        # === TEXT FILES ===
        if file_extension in ['txt', 'md', 'markdown']:
            content = file_content.decode('utf-8')

        # === PDF FILES ===
        elif file_extension == 'pdf':
            print(f"📄 Processing PDF: {file.filename}")
            content = await _extract_pdf_content(file_content, file.filename)

        # === WORD DOCUMENTS ===
        elif file_extension in ['docx', 'doc']:
            print(f"📄 Processing Word document: {file.filename}")
            content = await _extract_word_content(file_content, file.filename, file_extension)

        # === EPUB FILES ===
        elif file_extension == 'epub':
            print(f"📚 Processing EPUB: {file.filename}")
            content = await _extract_epub_content(file_content, file.filename)

        # === DATA FILES ===
        elif file_extension in ['csv', 'json']:
            content = file_content.decode('utf-8')

        else:
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

        print(f"✅ Document '{title}' saved to database (ID: {new_doc.id})")

        # Process through document manager
        success = await enhanced_rag_service.document_manager.add_document(db, new_doc.id)

        if not success:
            print(f"⚠️ Document saved but processing failed")
            raise HTTPException(status_code=400, detail="Document uploaded but processing failed")

        return new_doc

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to upload file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to upload file: {str(e)}")


# === EXTRACTION HELPERS ===

async def _extract_pdf_content(file_content: bytes, filename: str) -> str:
    """Extract text from PDF."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        content = "\n\n".join([page.page_content for page in pages])

        if not content.strip():
            return "[PDF processed but no text content extracted]"
        print(f"✅ Extracted {len(content):,} characters from PDF")
        return content

    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return f"[PDF extraction failed: {str(e)}]"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _extract_word_content(file_content: bytes, filename: str, ext: str) -> str:
    """Extract text from Word document."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{ext}', delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(temp_path)
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])

        if not content.strip():
            return "[Word document processed but no text content extracted]"
        print(f"✅ Extracted {len(content):,} characters from Word document")
        return content

    except Exception as e:
        print(f"❌ Error extracting Word document: {e}")
        return f"[Word extraction failed: {str(e)}]"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _extract_epub_content(file_content: bytes, filename: str) -> str:
    """
    Extract text from EPUB file.
    Uses ebooklib + BeautifulSoup (pure Python, no system dependencies).
    """
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.epub', delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(temp_path)

        # Get metadata
        book_title = None
        book_author = None
        try:
            t = book.get_metadata('DC', 'title')
            book_title = t[0][0] if t else None
            a = book.get_metadata('DC', 'creator')
            book_author = a[0][0] if a else None
        except:
            pass

        book_title = book_title or filename
        book_author = book_author or 'Unknown'
        print(f"📖 EPUB: {book_title} by {book_author}")

        # Get reading order
        spine_ids = [item[0] for item in book.spine]

        chapters = []
        chapter_num = 0

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            if item.get_id() not in spine_ids:
                continue

            # Parse HTML
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Remove non-content
            for elem in soup(['script', 'style', 'nav']):
                elem.decompose()

            # Find title
            title = None
            for tag in ['h1', 'h2']:
                elem = soup.find(tag)
                if elem:
                    t = elem.get_text(strip=True)
                    if t and 2 < len(t) < 200:
                        title = t
                        break

            # Extract text
            text_parts = []
            for tag in ['p', 'div', 'blockquote', 'li']:
                for element in soup.find_all(tag):
                    text = element.get_text(' ', strip=True)
                    if text and len(text) > 5:
                        text_parts.append(text)

            chapter_text = '\n\n'.join(text_parts)

            if len(chapter_text.strip()) < 50:
                continue

            chapter_num += 1
            chapter_title = title or f"Section {chapter_num}"
            chapters.append(f"=== {chapter_title} ===\n\n{chapter_text}")

        content = '\n\n'.join(chapters)

        if not content.strip():
            return "[EPUB processed but no text content extracted]"

        print(f"✅ Extracted {len(content):,} characters from {chapter_num} sections")
        return content

    except ImportError:
        # Fallback
        print("⚠️ ebooklib not available, trying UnstructuredEPubLoader...")
        try:
            from langchain_community.document_loaders import UnstructuredEPubLoader
            loader = UnstructuredEPubLoader(temp_path, mode="single")
            docs = loader.load()
            content = "\n\n".join([doc.page_content for doc in docs])
            if content.strip():
                return content
            return "[EPUB processed but no text content extracted]"
        except Exception as e:
            return f"[EPUB extraction failed: {str(e)}]"

    except Exception as e:
        print(f"❌ Error extracting EPUB: {e}")
        import traceback
        traceback.print_exc()
        return f"[EPUB extraction failed: {str(e)}]"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# === OTHER ROUTES (unchanged) ===

@router.delete("/all")
async def delete_all_documents(db: Session = Depends(get_db)):
    """Delete all documents."""
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
    """Delete a document (soft delete)."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    success = enhanced_rag_service.document_manager.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or deletion failed")
    return {"message": f"Document {document_id} successfully deleted"}


@router.post("/{document_id}/process")
async def process_document(document_id: int, db: Session = Depends(get_db)):
    """Process an uploaded document for RAG."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    success = await enhanced_rag_service.document_manager.add_document(db, document_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to process document")
    return {"message": "Document processed successfully"}


@router.post("/rebuild-index")
async def rebuild_index(db: Session = Depends(get_db)):
    """Rebuild the vector store index from scratch."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    try:
        print("🔄 Starting index rebuild...")
        success = await enhanced_rag_service.document_manager.rebuild_index(db)
        if not success:
            raise HTTPException(status_code=400, detail="Index rebuild failed")

        stats = enhanced_rag_service.document_manager.get_stats()
        return {"message": "Index rebuilt successfully", "stats": stats}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/{document_id}/status")
async def get_document_status(document_id: int, db: Session = Depends(get_db)):
    """Get the status of a document."""
    from app.services.enhanced_rag_service import enhanced_rag_service
    from app.database import LoreDocument

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
    return enhanced_rag_service.document_manager.get_stats()