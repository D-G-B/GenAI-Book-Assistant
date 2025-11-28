"""
Document Manager Service
========================

Coordinates the lifecycle of documents across the Database, Vector Store, and File Manifest.
This service handles:
1. Document Ingestion: processing text, PDF, and EPUB content.
2. Structure Analysis: Automatically detecting chapters, prologues, and epilogues.
3. Chunking: Splitting text into semantic units with metadata (chapter numbers) for spoiler protection.
4. Synchronization: Ensuring the SQL DB, Vector DB, and JSON Manifest stay in sync.
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
    # MANIFEST MANAGEMENT
    # =========================================================================

    def _load_manifest(self):
        """Load the manifest of processed documents from disk."""
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
        """Save the current state of processed documents to disk."""
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
        """Check if a document ID has already been processed."""
        return document_id in self.processed_documents

    # =========================================================================
    # DOCUMENT INGESTION WORKFLOW
    # =========================================================================

    async def add_document(self, db: Session, document_id: int) -> bool:
        """
        Main Entry Point: Add and process a document.

        1. Checks for soft-deleted status (restores if found).
        2. Checks if already processed.
        3. Fetches content from SQL Database.
        4. Detects structure and chunks content.
        5. Adds chunks to Vector Store.
        6. Updates Manifest.
        """
        # 1. Restore if previously soft-deleted
        if self.vector_store_manager.is_deleted(document_id):
            print(f"♻️  Document {document_id} was previously soft-deleted. Restoring...")

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

        # 2. Skip if already active
        if self.is_processed(document_id):
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
            if db_doc.content.startswith('[') and 'extraction failed' in db_doc.content:
                print(f"⚠️ Skipping document with failed extraction")
                return False

            # 4. Process and Chunk (Structure Detection happens here)
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

            # 6. Calculate True Max Chapter for Slider
            # We filter for 'body' sections that have a valid number.
            # This ensures the slider goes to the *highest* chapter (e.g. 60),
            # not just the *count* of chapters (which might be lower if some are skipped).
            body_chapters = [
                doc.metadata['chapter_number']
                for doc in chunks
                if doc.metadata.get('section_type') == 'body'
                and doc.metadata.get('chapter_number') is not None
            ]

            true_max_chapter = max(body_chapters) if body_chapters else 0

            # Fallback: If regex failed but chunks exist, use chunk count estimate or 50
            if true_max_chapter == 0 and len(chunks) > 0:
                true_max_chapter = 1

            # 7. Update Manifest
            self.processed_documents[document_id] = {
                'title': db_doc.title,
                'filename': db_doc.filename,
                'chunk_count': len(chunks),
                'total_chapters': true_max_chapter,
                'processed_at': datetime.now().isoformat()
            }

            self._save_manifest()

            print(f"✅ Document {document_id} fully processed and synced")
            print(f"   📖 Max Chapter Detected: {true_max_chapter}")
            return True

        except Exception as e:
            print(f"❌ Error adding document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # STRUCTURE DETECTION & CHUNKING LOGIC
    # =========================================================================

    async def _process_and_chunk(self, db_doc: LoreDocument) -> List[Document]:
        """
        Analyzes document content to detect its structure (Chapters, Parts, etc.).
        Decides whether to use Structured Chunking (for spoiler protection) or
        Standard Chunking (flat text).
        """
        base_metadata = {
            'document_id': db_doc.id,
            'document_title': db_doc.title,
            'source_type': db_doc.source_type or 'text'
        }

        content = db_doc.content

        # Define Regex Patterns for Universal Structure Detection
        # We look for at least 2 matches to confirm a pattern is valid.
        patterns = [
            # Pattern A: Test/Custom format (e.g. === Chapter 1 ===)
            r'===\s*(.+?)\s*===',

            # Pattern B: Standard headers (Chapter 1, Chapter One, Chapter IV)
            # (?im) flags enable Case-Insensitive and Multiline matching
            r'((?im)^chapter\s+[\d\w]+.*?$)',

            # Pattern C: Major Divisions (Part I, Book One)
            r'((?im)^part\s+[\d\w]+.*?$)',
            r'((?im)^book\s+[\d\w]+.*?$)',

            # Pattern D: Explicit Special Sections (Prologue, Epilogue only lines)
            r'((?im)^(?:prologue|epilogue|interlude|preface|introduction).*?$)'
        ]

        selected_pattern = None

        # Iterate patterns to find the best fit for this document
        for pattern in patterns:
            matches = re.findall(pattern, content)
            # Threshold: If we see at least 2 headers, we assume structure exists
            if len(matches) >= 2:
                print(f"   📖 Detected structure: {len(matches)} sections using pattern '{pattern}'")
                selected_pattern = pattern
                break

        # Branch 1: Structure Detected -> Use Smart Chunking
        if selected_pattern:
            return self._chunk_structured_content(content, base_metadata, selected_pattern)

        # Branch 2: No Structure -> Use Fallback Chunking
        # Note: Spoiler protection will likely be ineffective here (all content is visible)
        print("   ⚠️ No chapter structure detected, using standard chunking")
        try:
            initial_docs = self.document_processor.process_content(
                content, db_doc.filename, base_metadata
            )
        except Exception as e:
            # Emergency fallback if processor fails
            if len(content.strip()) > 20:
                initial_docs = [Document(page_content=content, metadata=base_metadata)]
            else:
                return []

        if not initial_docs:
            return []

        return self._chunk_documents(initial_docs)

    def _chunk_structured_content(self, content: str, base_metadata: Dict[str, Any], chapter_pattern: str) -> List[Document]:
        """
        Splits content by the detected chapter pattern and assigns metadata.
        Uses 'section_type' (frontmatter, body, backmatter) for robust spoiler logic.
        """
        final_chunks = []

        # re.split with capturing groups returns [text_before, header1, text1, header2, text2...]
        parts = re.split(chapter_pattern, content)

        i = 0

        # Handle Frontmatter (text before the first header)
        if parts and parts[0].strip():
            intro_chunks = self._create_chunks(
                parts[0].strip(), base_metadata,
                section_type='frontmatter',
                chapter_number=0,
                chapter_title="Frontmatter",
                is_reference=False
            )
            final_chunks.extend(intro_chunks)
            i = 1
        else:
            i = 1

        # Iterate through (Header, Content) pairs
        while i < len(parts) - 1:
            chapter_title = parts[i].strip()
            chapter_content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Skip empty sections (often caused by double line breaks)
            if not chapter_content or len(chapter_content) < 50:
                i += 2
                continue

            # --- SMART SECTION LOGIC ---
            title_lower = chapter_title.lower()
            current_section_type = 'body'
            current_chapter_num = None

            # Rule 1: Backmatter (Dangerous Spoilers)
            # Epilogues, Afterwords, Appendices are always spoilers.
            if any(x in title_lower for x in ['epilogue', 'afterword', 'appendix', 'glossary']):
                current_section_type = 'backmatter'
                current_chapter_num = None # Number is irrelevant, always hidden if spoiler mode is ON

            # Rule 2: Frontmatter (Always Safe)
            # Prologues, Intros are safe to read anytime.
            elif any(x in title_lower for x in ['prologue', 'introduction', 'preface']):
                current_section_type = 'frontmatter'
                current_chapter_num = 0

            # Rule 3: Body Chapters (Filtered by Slider)
            else:
                current_section_type = 'body'
                current_chapter_num = self._extract_chapter_number(chapter_title)

                # Fallback: just increment or use 0 if we can't parse the number
                if current_chapter_num is None:
                    current_chapter_num = 0

            # Check if this is a glossary/appendix (legacy flag, kept for compatibility)
            is_reference = current_section_type == 'backmatter'

            # Generate chunks for this section
            chapter_chunks = self._create_chunks(
                chapter_content, base_metadata,
                section_type=current_section_type,
                chapter_number=current_chapter_num,
                chapter_title=chapter_title,
                is_reference=is_reference
            )
            final_chunks.extend(chapter_chunks)
            i += 2

        return final_chunks

    def _extract_chapter_number(self, title: str) -> Optional[int]:
        """
        Extracts chapter number from a title string using multiple strategies.
        """
        title_lower = title.lower()

        # Strategy 1: Standard Patterns (Digits)
        patterns = [
            r'(?:chapter|part|book|ch\.?)\s*(\d+)',
            r'^(\d+)\.',
            r'^(\d+)\s*[-–—]',
        ]

        for pattern in patterns:
            match = re.search(pattern, title_lower)
            if match:
                return int(match.group(1))

        # Strategy 2: Roman Numerals
        roman_pattern = r'(?:chapter|part|book)\s+([ivxlc]+)(?:\s|$|:)'
        match = re.search(roman_pattern, title_lower)
        if match:
            return self._roman_to_int(match.group(1).upper())

        # Strategy 3: Word Numbers
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

    def _roman_to_int(self, s: str) -> Optional[int]:
        """Helper to convert standard Roman numerals to integers."""
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val.get(s[i], 0) > rom_val.get(s[i - 1], 0):
                int_val += rom_val.get(s[i], 0) - 2 * rom_val.get(s[i - 1], 0)
            else:
                int_val += rom_val.get(s[i], 0)
        return int_val if int_val > 0 else None

    def _create_chunks(
            self,
            content: str,
            base_metadata: Dict[str, Any],
            section_type: str,
            chapter_number: Optional[int],
            chapter_title: str,
            is_reference: bool = False
    ) -> List[Document]:
        """
        Splits a single section/chapter into smaller vector-ready chunks.
        Attaches 'section_type' for robust filtering.
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
                'chapter_title': chapter_title,
                'chapter_number': chapter_number,
                'section_type': section_type,  # Crucial for new filter logic
                'is_reference_material': is_reference # Kept for legacy support
            })

            documents.append(Document(
                page_content=chunk_text.strip(),
                metadata=chunk_metadata
            ))

        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Fallback chunker for documents where no chapter structure was detected.
        Processes the text as a flat stream with 'body' type.
        """
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
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'section_type': 'body', # Default to body for unstructured text
                    'chapter_number': None
                })

                final_documents.append(Document(
                    page_content=chunk.strip(),
                    metadata=chunk_metadata
                ))

        return final_documents

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