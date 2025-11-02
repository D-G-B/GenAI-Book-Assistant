"""
Document Manager - Coordinates document lifecycle across database, vector store, and manifest.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from sqlalchemy.orm import Session

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.database import LoreDocument
from app.services.vector_store_manager import VectorStoreManager
from app.services.advanced_document_loaders import document_processor


class DocumentManager:
    """
    Central coordinator for all document operations.
    Maintains synchronization between Database, Vector Store, and Manifest.
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        print("🚀 Initializing Document Manager...")

        self.vector_store_manager = vector_store_manager

        # Manifest path
        self.manifest_path = Path("./faiss_index/manifest.json")

        # Track processed documents: {document_id: metadata}
        self.processed_documents: Dict[int, Dict[str, Any]] = {}

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # Document processor for multi-format support
        self.document_processor = document_processor

        # Load manifest
        self._load_manifest()

        print("✅ Document Manager initialized")

    def _load_manifest(self):
        """Load manifest of processed documents."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)

                    # Load processed document IDs
                    processed_ids = data.get('processed_document_ids', [])
                    for doc_id in processed_ids:
                        self.processed_documents[doc_id] = {
                            'loaded_from_manifest': True
                        }

                    print(f"📋 Loaded manifest: {len(self.processed_documents)} documents")
            except Exception as e:
                print(f"⚠️ Could not load manifest: {e}")

    def _save_manifest(self):
        """Save manifest of processed documents."""
        try:
            self.manifest_path.parent.mkdir(exist_ok=True)

            manifest_data = {
                'processed_document_ids': list(self.processed_documents.keys()),
                'last_updated': datetime.now().isoformat(),
                'total_documents': len(self.processed_documents)
            }

            with open(self.manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)

            print(f"📋 Manifest saved: {len(self.processed_documents)} documents")
        except Exception as e:
            print(f"⚠️ Failed to save manifest: {e}")

    def is_processed(self, document_id: int) -> bool:
        """Check if a document has been processed."""
        return document_id in self.processed_documents

    async def add_document(self, db: Session, document_id: int) -> bool:
        """
        Add and process a document.

        This is the operation that:
        1. Validates document exists in database
        2. Processes and chunks the document
        3. Adds chunks to vector store
        4. Updates manifest

        Returns True if successful, False otherwise.
        """

        # Check if this document was previously soft-deleted
        if self.vector_store_manager.is_deleted(document_id):
            print(f"♻️  Document {document_id} was previously soft-deleted")
            print(f"✨ Simply restoring (toggling soft delete off)...")

            # Just remove from deleted set - chunks are already there!
            self.vector_store_manager.deleted_document_ids.discard(document_id)
            self.vector_store_manager._save_deleted_ids()

            # Add to processed documents tracking
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if db_doc:
                self.processed_documents[document_id] = {
                    'title': db_doc.title,
                    'filename': db_doc.filename,
                    'restored_at': datetime.now().isoformat()
                }
                self._save_manifest()

            print(f"✅ Document {document_id} restored (no reprocessing needed)")
            return True

        # Skip if already processed (and not deleted)
        if self.is_processed(document_id):
            print(f"⏭️  Document {document_id} already processed, skipping")
            return True

        try:
            # 1. Get document from database
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

            if not db_doc or not db_doc.content:
                print(f"❌ Document {document_id} not found or has no content")
                return False

            print(f"📄 Processing: {db_doc.title} ({db_doc.filename})")

            # Skip failed extractions
            if db_doc.content.startswith('[PDF') and 'extraction failed' in db_doc.content:
                print(f"⚠️ Skipping document with failed extraction")
                return False

            # 2. Process and chunk document
            chunks = await self._process_and_chunk(db_doc)

            if not chunks:
                print(f"❌ No valid chunks created from document {document_id}")
                return False

            print(f"✅ Created {len(chunks)} valid chunks")

            # 3. Add to vector store
            success = self.vector_store_manager.add_documents(chunks)

            if not success:
                print(f"❌ Failed to add chunks to vector store")
                return False

            # 4. Update manifest
            self.processed_documents[document_id] = {
                'title': db_doc.title,
                'filename': db_doc.filename,
                'chunk_count': len(chunks),
                'processed_at': datetime.now().isoformat()
            }

            self._save_manifest()

            print(f"✅ Document {document_id} fully processed and synced")
            return True

        except Exception as e:
            print(f"❌ Error adding document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _process_and_chunk(self, db_doc: LoreDocument) -> List[Document]:
        """Process document content and create chunks."""

        # Create base metadata
        base_metadata = {
            'document_id': db_doc.id,
            'document_title': db_doc.title,
            'source_type': db_doc.source_type or 'text'
        }

        # Use document processor for different file types
        try:
            initial_docs = self.document_processor.process_content(
                db_doc.content,
                db_doc.filename,
                base_metadata
            )
        except Exception as e:
            print(f"⚠️ Error with document processor: {e}")
            # Fallback to plain text
            if len(db_doc.content.strip()) > 20:
                initial_docs = [Document(page_content=db_doc.content, metadata=base_metadata)]
            else:
                return []

        if not initial_docs:
            return []

        # Split into chunks
        final_documents = []
        for document in initial_docs:
            # Skip empty content
            if not document.page_content or not document.page_content.strip():
                continue

            # Skip very short content
            if len(document.page_content.strip()) < 20:
                continue

            try:
                chunks = self.text_splitter.split_text(document.page_content)
            except Exception as e:
                print(f"⚠️ Error splitting text: {e}")
                continue

            for i, chunk in enumerate(chunks):
                # Validate chunk
                if not chunk or not chunk.strip() or len(chunk.strip()) < 10:
                    continue

                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })

                final_documents.append(Document(
                    page_content=chunk.strip(),
                    metadata=chunk_metadata
                ))

        return final_documents

    def delete_document(self, db: Session, document_id: int) -> bool:
        """
        Delete a document (soft delete).

        This is the operation that:
        1. Removes from database
        2. Soft-deletes from vector store
        3. Updates manifest

        Returns True if successful, False otherwise.
        """

        try:
            # 1. Get document from database
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

            if not db_doc:
                print(f"⚠️ Document {document_id} not found in database")
                return False

            doc_title = db_doc.title

            # 2. Delete from database
            db.delete(db_doc)
            db.commit()
            print(f"🗑️ Removed document {document_id} from database")

            # 3. Soft delete from vector store
            self.vector_store_manager.soft_delete_document(document_id)

            # 4. Update manifest
            if document_id in self.processed_documents:
                del self.processed_documents[document_id]
                self._save_manifest()

            print(f"✅ Document '{doc_title}' (ID: {document_id}) fully deleted")
            return True

        except Exception as e:
            print(f"❌ Error deleting document {document_id}: {e}")
            db.rollback()
            return False

    def delete_all_documents(self, db: Session) -> bool:
        """
        Delete all documents.

        This is the atomic operation that:
        1. Clears database
        2. Clears vector store
        3. Clears manifest
        """

        try:
            # 1. Clear database
            db.query(LoreDocument).delete()
            db.commit()
            print("🗑️ Cleared all documents from database")

            # 2. Clear vector store
            self.vector_store_manager.clear_all()

            # 3. Clear manifest
            self.processed_documents.clear()
            if self.manifest_path.exists():
                self.manifest_path.unlink()

            print("✅ All documents deleted from all systems")
            return True

        except Exception as e:
            print(f"❌ Error deleting all documents: {e}")
            db.rollback()
            return False

    async def rebuild_index(self, db: Session) -> bool:
        """
        Rebuild the vector store index from scratch.

        This physically removes soft-deleted documents and optimizes the index.
        """

        try:
            print("🔄 Starting index rebuild...")

            # Get all documents from database
            all_db_docs = db.query(LoreDocument).all()

            if not all_db_docs:
                print("⚠️ No documents in database to rebuild from")
                self.vector_store_manager.vector_store = None
                return True

            # Process all documents and collect chunks
            all_chunks = []

            for db_doc in all_db_docs:
                if not db_doc.content:
                    continue

                print(f"   Processing: {db_doc.title}")
                chunks = await self._process_and_chunk(db_doc)
                all_chunks.extend(chunks)

            # Rebuild vector store
            success = self.vector_store_manager.rebuild_index(all_chunks)

            if success:
                # Update manifest with all processed documents
                self.processed_documents = {
                    doc.id: {
                        'title': doc.title,
                        'filename': doc.filename,
                        'rebuilt_at': datetime.now().isoformat()
                    }
                    for doc in all_db_docs
                }
                self._save_manifest()

            return success

        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_document_status(self, document_id: int) -> Dict[str, Any]:
        """Get the status of a document across all systems."""

        return {
            'processed': document_id in self.processed_documents,
            'soft_deleted': self.vector_store_manager.is_deleted(document_id),
            'metadata': self.processed_documents.get(document_id, {})
        }

    def list_all_documents(self, db: Session, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """
        List all documents with their status.

        Args:
            db: Database session
            include_deleted: If True, includes soft-deleted documents with status
        """

        db_docs = db.query(LoreDocument).all()

        result = []
        for doc in db_docs:
            status = self.get_document_status(doc.id)

            # Skip soft-deleted unless requested
            if not include_deleted and status['soft_deleted']:
                continue

            result.append({
                'id': doc.id,
                'title': doc.title,
                'filename': doc.filename,
                'source_type': doc.source_type,
                'created_at': doc.created_at.isoformat() if doc.created_at else None,
                'processed': status['processed'],
                'soft_deleted': status['soft_deleted'],
                'chunk_count': status['metadata'].get('chunk_count', 0)
            })

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""

        vector_stats = self.vector_store_manager.get_stats()

        return {
            'processed_documents': len(self.processed_documents),
            'total_chunks': vector_stats['total_chunks'],
            'deleted_documents': vector_stats['deleted_documents'],
            'should_rebuild': vector_stats['should_rebuild'],
            'vector_store_exists': vector_stats['vector_store_exists']
        }