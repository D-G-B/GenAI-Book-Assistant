"""
Document Manager Service
========================

Coordinates the lifecycle of documents across the Database, Vector Store, and File Manifest.
This service handles:
1. Document Ingestion: processing text, PDF, and EPUB content.
2. Structure Analysis: Automatically detecting chapters via content scanning.
3. Chunking: Splitting text into semantic units with chapter_number for spoiler protection.
4. Synchronization: Ensuring the SQL DB, Vector DB, and JSON Manifest stay in sync.

SIMPLIFIED SPOILER MODEL:
- chapter_number: Integer (1, 2, 3...) for story chapters, None for reference material
- is_reference: Boolean - True for appendices, glossary, terminology, etc.
- Spoiler filter: Only return chunks where chapter_number <= max_chapter
- Reference toggle: Optionally include chunks where is_reference=True
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

        # Path to the JSON manifest that tracks processed files
        self.manifest_path = Path("./faiss_index/manifest.json")

        # In-memory track of processed documents: {document_id: metadata}
        self.processed_documents: Dict[int, Dict[str, Any]] = {}

        # Standard text splitter for chunking content within chapters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # Handles loading raw content from various file formats (PDF, EPUB, etc.)
        self.document_processor = document_processor

        # Load existing state
        self._load_manifest()

        print("✅ Document Manager initialized")

    # =========================================================================
    # MANIFEST MANAGEMENT (FIXED - now preserves full metadata)
    # =========================================================================

    def _load_manifest(self):
        """Load the manifest of processed documents from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)

                # NEW FORMAT: full metadata preserved
                if 'documents' in data:
                    self.processed_documents = {
                        int(k): v for k, v in data['documents'].items()
                    }
                    print(f"📋 Loaded manifest: {len(self.processed_documents)} documents (full metadata)")
                # LEGACY FORMAT: just IDs (migrate to new format)
                elif 'processed_document_ids' in data:
                    processed_ids = data.get('processed_document_ids', [])
                    for doc_id in processed_ids:
                        self.processed_documents[doc_id] = {
                            'migrated_from_legacy': True,
                            'chunk_count': 0,
                            'total_chapters': None
                        }
                    print(
                        f"📋 Loaded legacy manifest: {len(self.processed_documents)} documents (needs rebuild for metadata)")

            except Exception as e:
                print(f"⚠️ Could not load manifest: {e}")

    def _save_manifest(self):
        """Save the current state of processed documents to disk (FULL METADATA)."""
        try:
            self.manifest_path.parent.mkdir(exist_ok=True)

            manifest_data = {
                'version': 2,  # New format version
                'documents': {
                    str(k): v for k, v in self.processed_documents.items()
                },
                'last_updated': datetime.now().isoformat(),
                'total_documents': len(self.processed_documents)
            }

            with open(self.manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)

            print(f"📋 Manifest saved: {len(self.processed_documents)} documents (full metadata)")
        except Exception as e:
            print(f"⚠️ Failed to save manifest: {e}")

    def is_processed(self, document_id: int) -> bool:
        """Check if a document ID has already been processed."""
        return document_id in self.processed_documents

    # =========================================================================
    # DOCUMENT INGESTION WORKFLOW
    # =========================================================================

    async def add_document(self, db: Session, document_id: int) -> bool:
        """
        Main Entry Point: Add and process a document.
        """
        # 1. Restore if previously soft-deleted
        if self.vector_store_manager.is_deleted(document_id):
            print(f"♻️  Document {document_id} was previously soft-deleted. Restoring...")

            self.vector_store_manager.deleted_document_ids.discard(document_id)
            self.vector_store_manager._save_deleted_ids()

            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if db_doc:
                # Preserve existing metadata if available
                existing = self.processed_documents.get(document_id, {})
                existing['restored_at'] = datetime.now().isoformat()
                self.processed_documents[document_id] = existing
                self._save_manifest()

            print(f"✅ Document {document_id} restored")
            return True

        # 2. Skip if already active (but check if metadata is complete)
        if self.is_processed(document_id):
            existing = self.processed_documents.get(document_id, {})
            # If we have real metadata, skip; if migrated from legacy, allow reprocess
            if not existing.get('migrated_from_legacy'):
                print(f"⏭️  Document {document_id} already processed, skipping")
                return True

        try:
            # 3. Fetch from DB
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()

            if not db_doc or not db_doc.content:
                print(f"❌ Document {document_id} not found or has no content")
                return False

            print(f"📄 Processing: {db_doc.title} ({db_doc.filename})")

            # Check for extraction errors
            if db_doc.content.startswith('[') and 'extraction failed' in db_doc.content.lower():
                print(f"⚠️ Skipping document with failed extraction")
                return False

            # 4. Process and Chunk
            chunks = await self._process_and_chunk(db_doc)

            if not chunks:
                print(f"❌ No valid chunks created from document {document_id}")
                return False

            print(f"✅ Created {len(chunks)} chunks")

            # 5. Add to Vector Store
            success = self.vector_store_manager.add_documents(chunks)

            if not success:
                print(f"❌ Failed to add chunks to vector store")
                return False

            # 6. Calculate max chapter from body chunks
            body_chapters = [
                doc.metadata['chapter_number']
                for doc in chunks
                if doc.metadata.get('chapter_number') is not None
                   and not doc.metadata.get('is_reference', False)
            ]

            max_chapter = max(body_chapters) if body_chapters else 0

            # Count reference chunks
            reference_chunks = sum(1 for doc in chunks if doc.metadata.get('is_reference', False))

            # 7. Update Manifest with FULL metadata
            self.processed_documents[document_id] = {
                'title': db_doc.title,
                'filename': db_doc.filename,
                'chunk_count': len(chunks),
                'total_chapters': max_chapter,
                'reference_chunks': reference_chunks,
                'processed_at': datetime.now().isoformat()
            }

            self._save_manifest()

            print(f"✅ Document {document_id} fully processed and synced")
            print(f"   📖 Max Chapter: {max_chapter}, Reference chunks: {reference_chunks}")
            return True

        except Exception as e:
            print(f"❌ Error adding document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # CHAPTER DETECTION (IMPROVED)
    # =========================================================================

    def _detect_chapters_in_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Scan content to find chapter markers and their positions.
        Returns list of: {'start': int, 'title': str, 'chapter_number': int|None, 'is_reference': bool}
        """
        chapters = []

        # Patterns to find chapter markers
        # ORDER MATTERS: The first pattern to match a region "wins".
        patterns = [
            # 1. AUTHOR INTENT (Highest Priority)
            # We look for what the author wrote first. If we find "Chapter 1",
            # we will ignore any "=== Section ===" markers that appear nearby.
            (r'^Chapter\s+(\d+)\b', 'numbered'),
            (r'^CHAPTER\s+(\d+)\b', 'numbered'),
            (r'^Chapter\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty)\b',
             'word'),
            (r'^Chapter\s+([IVXLC]+)\b', 'roman'),

            # 2. MACHINE ARTIFACTS (Fallback)
            # We only use these if the Author patterns didn't find anything at this position.

            # "If we can't find a real chapter title above, use the file section number, but keep it mathematically useful so the slider still works."
            (r'===\s*Section\s+(\d+)\s*===', 'numbered'),

            # Existing generic patterns
            (r'===\s*(?:Chapter\s+)?(\d+)\s*===', 'numbered'),
            (r'===\s*(.+?)\s*===', 'titled'),

            # Book divisions
            (r'^Book\s+(?:One|Two|Three|Four|Five|I|II|III|IV|V)\s*[-:]\s*(.+)$', 'book_division'),
        ]

        # Reference section markers
        reference_patterns = [
            r'(?:^|\n)(?:Appendix|APPENDIX)\s*[IVXLC\d]*\s*[-:]?\s*(.+)?',
            r'(?:^|\n)(?:Glossary|GLOSSARY|Terminology|TERMINOLOGY)',
            r'(?:^|\n)(?:Afterword|AFTERWORD|Epilogue|EPILOGUE)',
            r'(?:^|\n)(?:Notes|NOTES|Bibliography|BIBLIOGRAPHY)',
            r'(?:^|\n)(?:Cartographic|CARTOGRAPHIC|Map|MAP)',
            r'(?:^|\n)(?:About the Author|ABOUT THE AUTHOR)',
        ]

        # Word to number mapping
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }

        # First, find all chapter markers
        found_positions = set()

        for pattern, pattern_type in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                start = match.start()

                # Skip if we already found something at this position
                if any(abs(start - pos) < 20 for pos in found_positions):
                    continue

                found_positions.add(start)
                title = match.group(0).strip()
                chapter_num = None

                if pattern_type == 'numbered':
                    chapter_num = int(match.group(1))
                elif pattern_type == 'word':
                    word = match.group(1).lower()
                    chapter_num = word_to_num.get(word)
                elif pattern_type == 'roman':
                    chapter_num = self._roman_to_int(match.group(1).upper())
                elif pattern_type == 'book_division':
                    # Book divisions aren't chapters, mark as structural
                    chapter_num = None

                chapters.append({
                    'start': start,
                    'title': title,
                    'chapter_number': chapter_num,
                    'is_reference': False
                })

        # Find reference sections
        for pattern in reference_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                start = match.start()

                if any(abs(start - pos) < 20 for pos in found_positions):
                    continue

                found_positions.add(start)
                chapters.append({
                    'start': start,
                    'title': match.group(0).strip(),
                    'chapter_number': None,
                    'is_reference': True
                })

        # Sort by position
        chapters.sort(key=lambda x: x['start'])

        return chapters

    def _roman_to_int(self, s: str) -> Optional[int]:
        """Convert Roman numerals to integer."""
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val.get(s[i], 0) > rom_val.get(s[i - 1], 0):
                int_val += rom_val.get(s[i], 0) - 2 * rom_val.get(s[i - 1], 0)
            else:
                int_val += rom_val.get(s[i], 0)
        return int_val if int_val > 0 else None

    # =========================================================================
    # PROCESSING & CHUNKING
    # =========================================================================

    async def _process_and_chunk(self, db_doc: LoreDocument) -> List[Document]:
        """
        Process document content and create chunks with chapter metadata.
        """
        base_metadata = {
            'document_id': db_doc.id,
            'document_title': db_doc.title,
            'source_type': db_doc.source_type or 'text'
        }

        content = db_doc.content

        # Detect chapter structure
        chapters = self._detect_chapters_in_content(content)

        if len(chapters) >= 2:
            print(f"   📖 Detected {len(chapters)} sections in content")
            return self._chunk_with_chapters(content, chapters, base_metadata)
        else:
            print("   ⚠️ No chapter structure detected, using flat chunking")
            return self._chunk_flat(content, base_metadata)

    def _chunk_with_chapters(
            self,
            content: str,
            chapters: List[Dict[str, Any]],
            base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Chunk content using detected chapter boundaries.
        """
        final_chunks = []

        # Handle content before first chapter (frontmatter)
        if chapters and chapters[0]['start'] > 100:
            frontmatter = content[:chapters[0]['start']].strip()
            if len(frontmatter) > 50:
                chunks = self._create_chunks(
                    frontmatter, base_metadata,
                    chapter_number=None,
                    chapter_title="Frontmatter",
                    is_reference=False
                )
                final_chunks.extend(chunks)

        # Process each chapter
        for i, chapter in enumerate(chapters):
            start = chapter['start']
            end = chapters[i + 1]['start'] if i + 1 < len(chapters) else len(content)

            chapter_content = content[start:end].strip()

            # Skip very short sections
            if len(chapter_content) < 100:
                continue

            chunks = self._create_chunks(
                chapter_content, base_metadata,
                chapter_number=chapter['chapter_number'],
                chapter_title=chapter['title'],
                is_reference=chapter['is_reference']
            )
            final_chunks.extend(chunks)

        return final_chunks

    def _chunk_flat(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """
        Fallback: chunk content without chapter structure.
        All chunks get chapter_number=1 so spoiler slider works (at minimum).
        """
        chunks = self.text_splitter.split_text(content)
        documents = []

        for i, chunk_text in enumerate(chunks):
            if not chunk_text or len(chunk_text.strip()) < 20:
                continue

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chapter_number': 1,  # Default to chapter 1 for unstructured
                'chapter_title': 'Content',
                'is_reference': False
            })

            documents.append(Document(
                page_content=chunk_text.strip(),
                metadata=chunk_metadata
            ))

        return documents

    def _create_chunks(
            self,
            content: str,
            base_metadata: Dict[str, Any],
            chapter_number: Optional[int],
            chapter_title: str,
            is_reference: bool
    ) -> List[Document]:
        """
        Split a section into smaller chunks with consistent metadata.
        """
        chunks = self.text_splitter.split_text(content)
        documents = []

        for i, chunk_text in enumerate(chunks):
            if not chunk_text or len(chunk_text.strip()) < 20:
                continue

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chapter_number': chapter_number,
                'chapter_title': chapter_title,
                'is_reference': is_reference
            })

            documents.append(Document(
                page_content=chunk_text.strip(),
                metadata=chunk_metadata
            ))

        return documents

    # =========================================================================
    # DELETION & MAINTENANCE
    # =========================================================================

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
                self.processed_documents.clear()
                self._save_manifest()
                return True

            # Clear existing
            self.vector_store_manager.clear_all()
            self.processed_documents.clear()

            # Reprocess each document
            for db_doc in all_db_docs:
                if not db_doc.content:
                    continue

                print(f"   Processing: {db_doc.title}")
                chunks = await self._process_and_chunk(db_doc)

                if chunks:
                    self.vector_store_manager.add_documents(chunks)

                    # Calculate stats
                    body_chapters = [
                        doc.metadata['chapter_number']
                        for doc in chunks
                        if doc.metadata.get('chapter_number') is not None
                           and not doc.metadata.get('is_reference', False)
                    ]
                    max_chapter = max(body_chapters) if body_chapters else 0
                    reference_chunks = sum(1 for doc in chunks if doc.metadata.get('is_reference', False))

                    self.processed_documents[db_doc.id] = {
                        'title': db_doc.title,
                        'filename': db_doc.filename,
                        'chunk_count': len(chunks),
                        'total_chapters': max_chapter,
                        'reference_chunks': reference_chunks,
                        'rebuilt_at': datetime.now().isoformat()
                    }

            self._save_manifest()
            print(f"✅ Index rebuilt: {len(self.processed_documents)} documents")
            return True

        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_document_status(self, document_id: int) -> Dict[str, Any]:
        """Get the status of a specific document across all systems."""
        return {
            'processed': document_id in self.processed_documents,
            'soft_deleted': self.vector_store_manager.is_deleted(document_id),
            'metadata': self.processed_documents.get(document_id, {})
        }

    def list_all_documents(self, db: Session, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """List all documents in the system with their processing status."""
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
                'total_chapters': status['metadata'].get('total_chapters'),
                'reference_chunks': status['metadata'].get('reference_chunks', 0)
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
