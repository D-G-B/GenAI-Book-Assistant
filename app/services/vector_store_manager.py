"""
Vector Store Manager
====================

Handles all interactions with the FAISS vector database.
This includes:
1. Persistence: Loading and saving the index to disk.
2. Soft Deletion: Managing a 'deleted' list to hide documents without expensive rebuilding.
3. Retrieval: Providing a powerful retriever with support for metadata filtering.
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStoreManager:
    """Manages FAISS vector store with soft delete and spoiler filtering support."""

    def __init__(self, persist_path: str = "./faiss_index"):
        self.persist_path = Path(persist_path)
        self.deleted_ids_path = self.persist_path / "deleted_ids.json"

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        # Track soft-deleted document IDs
        self.deleted_document_ids: Set[int] = set()

        # The actual vector store
        self.vector_store: Optional[FAISS] = None

        # Load existing state
        self._load_deleted_ids()
        self._load_vector_store()

        print(f"✅ Vector Store Manager initialized")
        if self.deleted_document_ids:
            print(f"   📋 Tracking {len(self.deleted_document_ids)} soft-deleted documents")

    def _load_vector_store(self):
        """Load FAISS vector store from disk if it exists."""
        if self.persist_path.exists():
            try:
                print(f"📂 Loading vector store from {self.persist_path}")
                self.vector_store = FAISS.load_local(
                    str(self.persist_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                try:
                    chunk_count = self.vector_store.index.ntotal
                    print(f"✅ Loaded vector store with {chunk_count} chunks")
                except:
                    print(f"✅ Loaded vector store")

            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
                self.vector_store = None
        else:
            print("📝 No existing vector store found, will create new one")

    def _load_deleted_ids(self):
        """Load soft-deleted document IDs from disk."""
        if self.deleted_ids_path.exists():
            try:
                with open(self.deleted_ids_path, 'r') as f:
                    data = json.load(f)
                    self.deleted_document_ids = set(data.get('deleted_ids', []))
                    print(f"📋 Loaded {len(self.deleted_document_ids)} deleted document IDs")
            except Exception as e:
                print(f"⚠️ Could not load deleted IDs: {e}")
                self.deleted_document_ids = set()

    def _save_deleted_ids(self):
        """Save soft-deleted document IDs to disk."""
        try:
            self.persist_path.mkdir(exist_ok=True)

            data = {
                'deleted_ids': list(self.deleted_document_ids),
                'last_updated': datetime.now().isoformat()
            }

            with open(self.deleted_ids_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"💾 Saved deleted IDs: {len(self.deleted_document_ids)} documents")
        except Exception as e:
            print(f"⚠️ Failed to save deleted IDs: {e}")

    def save_to_disk(self):
        """Save vector store and deleted IDs to disk."""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(str(self.persist_path))
                self._save_deleted_ids()
                print(f"💾 Vector store saved to {self.persist_path}")
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        if not documents:
            return False

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                print(f"✅ Created vector store with {len(documents)} chunks")
            else:
                success_count = 0
                for doc in documents:
                    try:
                        self.vector_store.add_documents([doc])
                        success_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to add chunk: {e}")
                        continue

                if success_count > 0:
                    print(f"✅ Added {success_count}/{len(documents)} chunks to vector store")
                else:
                    print(f"❌ Failed to add any chunks")
                    return False

            self.save_to_disk()
            return True

        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            return False

    def soft_delete_document(self, document_id: int):
        """Mark a document as deleted (soft delete)."""
        self.deleted_document_ids.add(document_id)
        self._save_deleted_ids()
        print(f"🗑️ Soft-deleted document ID: {document_id}")

    def undelete_document(self, document_id: int):
        """Restore a soft-deleted document."""
        if document_id in self.deleted_document_ids:
            self.deleted_document_ids.remove(document_id)
            self._save_deleted_ids()
            print(f"♻️ Restored document ID: {document_id}")

    def is_deleted(self, document_id: int) -> bool:
        """Check if a document is soft-deleted."""
        return document_id in self.deleted_document_ids

    def get_retriever(
        self,
        k: int = 8,  # Increased context window for better retrieval
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None
    ):
        """
        Get a retriever with filtering support.

        Args:
            k: Number of results to return
            document_id: Optional - filter to specific document
            max_chapter: Optional - filter to chapters <= this number (spoiler protection)
        """
        if self.vector_store is None:
            return None

        def filter_function(metadata: Dict[str, Any]) -> bool:
            """
            Filter based on document_id, deleted status, and Section Type.
            """
            doc_id = metadata.get("document_id")

            # 1. Filter out soft-deleted documents
            if doc_id in self.deleted_document_ids:
                return False

            # 2. Filter to specific document if requested
            if document_id is not None:
                if doc_id != document_id:
                    return False

            # 3. Spoiler Protection Filter
            if max_chapter is not None:
                sec_type = metadata.get("section_type", "body") # Default to body
                ch_num = metadata.get("chapter_number")

                # A. Always ALLOW Frontmatter (Prologue, Intro)
                if sec_type == 'frontmatter':
                    return True

                # B. Always BLOCK Backmatter (Epilogue, Appendices)
                # If spoiler mode is ON, these are hidden.
                if sec_type == 'backmatter':
                    return False

                # C. Filter BODY by Chapter Number
                if sec_type == 'body':
                    # If regex failed to find a number, assume it's safe (or dangerous? Safe is better for UX)
                    if ch_num is None:
                        return True
                    return ch_num <= max_chapter

            return True

        # Increase fetch_k when filtering to ensure we get enough results
        fetch_k = k * 4 if (document_id or max_chapter) else k * 2

        search_kwargs = {
            "k": k,
            "fetch_k": fetch_k,
            "filter": filter_function
        }

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def rebuild_index(self, all_documents: List[Document]) -> bool:
        """Rebuild the vector store from scratch."""
        print("🔄 Rebuilding vector store index...")

        try:
            active_docs = [
                doc for doc in all_documents
                if doc.metadata.get("document_id") not in self.deleted_document_ids
            ]

            if not active_docs:
                print("⚠️ No active documents to rebuild index")
                self.vector_store = None
                return True

            self.vector_store = FAISS.from_documents(active_docs, self.embeddings)

            old_deleted_count = len(self.deleted_document_ids)
            self.deleted_document_ids.clear()

            self.save_to_disk()

            chunk_count = self.vector_store.index.ntotal
            print(f"✅ Index rebuilt: {chunk_count} chunks")
            print(f"   🗑️ Physically removed {old_deleted_count} soft-deleted documents")

            return True

        except Exception as e:
            print(f"❌ Failed to rebuild index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def should_rebuild(self, threshold: float = 0.2) -> bool:
        """Check if index should be rebuilt based on ratio of deleted documents."""
        if not self.vector_store:
            return False

        try:
            total_chunks = self.vector_store.index.ntotal
            deleted_count = len(self.deleted_document_ids)

            if total_chunks == 0:
                return False

            deleted_ratio = deleted_count / (deleted_count + 10)
            return deleted_ratio > threshold
        except:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "total_chunks": 0,
            "deleted_documents": len(self.deleted_document_ids),
            "vector_store_exists": self.vector_store is not None,
            "should_rebuild": False
        }

        if self.vector_store:
            try:
                stats["total_chunks"] = self.vector_store.index.ntotal
                stats["should_rebuild"] = self.should_rebuild()
            except:
                pass

        return stats

    def clear_all(self):
        """Clear the entire vector store and all tracking."""
        import shutil

        self.vector_store = None
        self.deleted_document_ids.clear()

        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
            print("🗑️ Cleared all vector store data")