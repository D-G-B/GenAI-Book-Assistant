"""
Document Manager - Coordinates document lifecycle across database, vector store, and manifest.
Now extracts chapter numbers for spoiler filtering.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re
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
        """
        # Check if previously soft-deleted
        if self.vector_store_manager.is_deleted(document_id):
            print(f"♻️  Document {document_id} was previously soft-deleted")
            print(f"✨ Restoring...")

            self.vector_store_manager.deleted_document_ids.discard(document_id)
            self.vector_store_manager._save_deleted_ids()

            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if db_doc:
                self.processed_documents[document_id] = {
                    'title': db_doc.title,
                    'filename': db_doc.filename,
                    'restored_at': datetime.now().isoformat()
                }
                self._save_manifest()

            print(f"✅ Document {document_id} restored")
            return True

        # Skip if already processed
        if self.is_processed(document_id):
            print(f"⏭️  Document {document_id} already processed, skipping")
            return True

        try:
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

            if not db_doc or not db_doc.content:
                print(f"❌ Document {document_id} not found or has no content")
                return False

            print(f"📄 Processing: {db_doc.title} ({db_doc.filename})")

            # Skip failed extractions
            if db_doc.content.startswith('[') and 'extraction failed' in db_doc.content:
                print(f"⚠️ Skipping document with failed extraction")
                return False

            # Process and chunk
            chunks = await self._process_and_chunk(db_doc)

            if not chunks:
                print(f"❌ No valid chunks created from document {document_id}")
                return False

            print(f"✅ Created {len(chunks)} chunks")

            # Add to vector store
            success = self.vector_store_manager.add_documents(chunks)

            if not success:
                print(f"❌ Failed to add chunks to vector store")
                return False

            # Count chapters
            chapter_numbers = set()
            for chunk in chunks:
                ch_num = chunk.metadata.get('chapter_number')
                if ch_num is not None:
                    chapter_numbers.add(ch_num)

            # Update manifest
            self.processed_documents[document_id] = {
                'title': db_doc.title,
                'filename': db_doc.filename,
                'chunk_count': len(chunks),
                'total_chapters': len(chapter_numbers) if chapter_numbers else None,
                'processed_at': datetime.now().isoformat()
            }

            self._save_manifest()

            print(f"✅ Document {document_id} fully processed and synced")
            if chapter_numbers:
                print(f"   📖 {len(chapter_numbers)} chapters detected")
            return True

        except Exception as e:
            print(f"❌ Error adding document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _process_and_chunk(self, db_doc: LoreDocument) -> List[Document]:
        """Process document content and detect structure automatically."""

        base_metadata = {
            'document_id': db_doc.id,
            'document_title': db_doc.title,
            'source_type': db_doc.source_type or 'text'
        }

        content = db_doc.content

        # 1. Define Chapter Patterns
        # A list of regex patterns to detect different book structures
        patterns = [
            # Pattern A: Your Test format (=== Chapter 1 ===)
            r'===\s*(.+?)\s*===',

            # Pattern B: Standard Books (Chapter 1: The Beginning)
            # (?im) = case-insensitive, multiline
            r'((?im)^chapter\s+\d+.*?$)',
            r'((?im)^chapter\s+[a-z]+.*?$)',

            # Pattern C: Parts (Part I)
            r'((?im)^part\s+\d+.*?$)'
        ]

        # 2. Detect which pattern fits this document
        selected_pattern = None

        # If it's explicitly an EPUB, default to Pattern A (or whatever your epub loader produces)
        # But for text/pdf, scan the content
        for pattern in patterns:
            # If we find at least 3 matches, assume this is the correct structure
            matches = re.findall(pattern, content)
            if len(matches) >= 3:
                print(f"   📖 Detected structure: {len(matches)} chapters using pattern '{pattern}'")
                selected_pattern = pattern
                break

        # 3. Use structured chunking if a pattern was found
        if selected_pattern:
            return self._chunk_structured_content(content, base_metadata, selected_pattern)

        # 4. Fallback to standard processing (no chapters)
        print("   ⚠️ No chapter structure detected, using standard chunking")
        try:
            initial_docs = self.document_processor.process_content(
                content, db_doc.filename, base_metadata
            )
        except Exception as e:
            if len(content.strip()) > 20:
                initial_docs = [Document(page_content=content, metadata=base_metadata)]
            else:
                return []

        if not initial_docs:
            return []

        return self._chunk_documents(initial_docs)

    def _chunk_structured_content(self, content: str, base_metadata: Dict[str, Any], chapter_pattern: str) -> List[
        Document]:
        """
        Universal chunker for any document with detectable chapters.
        Replaces the old _chunk_epub_content method.
        """
        final_chunks = []

        # Split content using the detected pattern
        parts = re.split(chapter_pattern, content)

        current_chapter_num = 0
        i = 0

        # Handle frontmatter (content before first chapter)
        if parts and parts[0].strip():
            intro_chunks = self._create_chunks(
                parts[0].strip(), base_metadata,
                chapter_number=0, chapter_title="Frontmatter", is_reference=False
            )
            final_chunks.extend(intro_chunks)
            i = 1
        else:
            i = 1

        # Process pairs: (Header, Content)
        # re.split includes the capturing groups (the headers) in the result list
        while i < len(parts) - 1:
            chapter_title = parts[i].strip()
            chapter_content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Skip empty sections
            if not chapter_content or len(chapter_content) < 50:
                i += 2
                continue

            # Extract number
            chapter_num = self._extract_chapter_number(chapter_title)

            if chapter_num is not None:
                current_chapter_num = chapter_num
            else:
                current_chapter_num += 1

            is_reference = self._is_reference_section(chapter_title)

            chapter_chunks = self._create_chunks(
                chapter_content,
                base_metadata,
                chapter_number=current_chapter_num if not is_reference else None,
                chapter_title=chapter_title,
                is_reference=is_reference
            )

            final_chunks.extend(chapter_chunks)
            i += 2

        return final_chunks

    def _extract_chapter_number(self, title: str) -> Optional[int]:
        """
        Extract chapter number from a title string.

        Handles:
        - "Chapter 1", "Chapter 01", "Chapter 123"
        - "Chapter One", "Chapter Two" (word numbers)
        - "Ch. 5", "Ch 5"
        """
        title_lower = title.lower()

        # Try numeric patterns first
        patterns = [
            r'chapter\s+(\d+)',  # Chapter 1, Chapter 01
            r'ch\.?\s*(\d+)',  # Ch. 5, Ch 5
            r'^(\d+)\.',  # 1. Title
            r'^(\d+)\s*[-–—]',  # 1 - Title
        ]

        for pattern in patterns:
            match = re.search(pattern, title_lower)
            if match:
                return int(match.group(1))

        # Try word numbers
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20
        }

        for word, num in word_to_num.items():
            if f'chapter {word}' in title_lower:
                return num

        return None

    def _is_reference_section(self, title: str) -> bool:
        """Check if a section title indicates reference material."""
        title_lower = title.lower()

        reference_markers = [
            'appendix', 'glossary', 'terminology', 'lexicon',
            'dramatis personae', 'cast of characters', 'index',
            'bibliography', 'afterword', 'about the author',
            'cartographic', 'map', 'timeline', 'chronology',
            'pronunciation', 'notes', 'acknowledgment'
        ]

        return any(marker in title_lower for marker in reference_markers)

    def _create_chunks(
            self,
            content: str,
            base_metadata: Dict[str, Any],
            chapter_number: Optional[int],
            chapter_title: str,
            is_reference: bool
    ) -> List[Document]:
        """Create chunks from content with proper metadata."""

        chunks = self.text_splitter.split_text(content)
        documents = []

        for i, chunk_text in enumerate(chunks):
            if not chunk_text or len(chunk_text.strip()) < 20:
                continue

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chapter_title': chapter_title,
                'is_reference_material': is_reference
            })

            # Only add chapter_number for non-reference material
            if chapter_number is not None and not is_reference:
                chunk_metadata['chapter_number'] = chapter_number

            documents.append(Document(
                page_content=chunk_text.strip(),
                metadata=chunk_metadata
            ))

        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Standard chunking for non-EPUB documents."""
        final_documents = []

        for document in documents:
            if not document.page_content or not document.page_content.strip():
                continue

            if len(document.page_content.strip()) < 20:
                continue

            try:
                chunks = self.text_splitter.split_text(document.page_content)
            except Exception as e:
                print(f"⚠️ Error splitting text: {e}")
                continue

            for i, chunk in enumerate(chunks):
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
        """Delete a document (soft delete)."""
        try:
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

            if not db_doc:
                print(f"⚠️ Document {document_id} not found in database")
                return False

            doc_title = db_doc.title

            db.delete(db_doc)
            db.commit()
            print(f"🗑️ Removed document {document_id} from database")

            self.vector_store_manager.soft_delete_document(document_id)

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
        """Delete all documents."""
        try:
            db.query(LoreDocument).delete()
            db.commit()
            print("🗑️ Cleared all documents from database")

            self.vector_store_manager.clear_all()

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
        """Rebuild the vector store index from scratch."""
        try:
            print("🔄 Starting index rebuild...")

            all_db_docs = db.query(LoreDocument).all()

            if not all_db_docs:
                print("⚠️ No documents in database to rebuild from")
                self.vector_store_manager.vector_store = None
                return True

            all_chunks = []

            for db_doc in all_db_docs:
                if not db_doc.content:
                    continue

                print(f"   Processing: {db_doc.title}")
                chunks = await self._process_and_chunk(db_doc)
                all_chunks.extend(chunks)

            success = self.vector_store_manager.rebuild_index(all_chunks)

            if success:
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
        """List all documents with their status."""
        db_docs = db.query(LoreDocument).all()

        result = []
        for doc in db_docs:
            status = self.get_document_status(doc.id)

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
                'chunk_count': status['metadata'].get('chunk_count', 0),
                'total_chapters': status['metadata'].get('total_chapters')
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
