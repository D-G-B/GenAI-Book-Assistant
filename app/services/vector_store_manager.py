"""
Vector Store Manager
====================

Handles all interactions with the FAISS vector database.

SIMPLIFIED SPOILER MODEL:
- Filter by chapter_number <= max_chapter
- Optionally include reference material (is_reference=True)
- No complex section_type logic
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)


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

        # Optional cross-encoder reranker (lazy-loaded on first use)
        self.reranker = self._init_reranker()

        # Load existing state
        self._load_deleted_ids()
        self._load_vector_store()

        logger.info("Vector Store Manager initialized")
        if self.deleted_document_ids:
            logger.info("Tracking %d soft-deleted documents", len(self.deleted_document_ids))

    @staticmethod
    def _init_reranker():
        if not settings.RERANKER_ENABLED:
            logger.info("Reranker disabled via settings")
            return None
        from app.services.reranker import CrossEncoderReranker
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL)

    def _load_vector_store(self):
        """Load FAISS vector store from disk if it exists."""
        if self.persist_path.exists():
            try:
                logger.info("Loading vector store from %s", self.persist_path)
                self.vector_store = FAISS.load_local(
                    str(self.persist_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                try:
                    chunk_count = self.vector_store.index.ntotal
                    logger.info("Loaded vector store with %d chunks", chunk_count)
                except AttributeError:
                    logger.info("Loaded vector store (chunk count unavailable)")

            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Could not load vector store: %s", e)
                self.vector_store = None
        else:
            logger.info("No existing vector store found, will create new one")

    def _load_deleted_ids(self):
        """Load soft-deleted document IDs from disk."""
        if self.deleted_ids_path.exists():
            try:
                with open(self.deleted_ids_path, 'r') as f:
                    data = json.load(f)
                    self.deleted_document_ids = set(data.get('deleted_ids', []))
                    logger.info("Loaded %d deleted document IDs", len(self.deleted_document_ids))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not load deleted IDs: %s", e)
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

            logger.info("Saved deleted IDs: %d documents", len(self.deleted_document_ids))
        except OSError as e:
            logger.warning("Failed to save deleted IDs: %s", e)

    def save_to_disk(self):
        """Save vector store and deleted IDs to disk."""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(str(self.persist_path))
                self._save_deleted_ids()
                logger.info("Vector store saved to %s", self.persist_path)
            except (OSError, RuntimeError) as e:
                logger.warning("Failed to save vector store: %s", e)

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        if not documents:
            return False

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info("Created vector store with %d chunks", len(documents))
            else:
                # Batched: a single add_documents call lets the embedding model
                # run sentence-transformers' internal mini-batching once over
                # the whole list instead of once per chunk. ~10-50x faster on
                # CPU for multi-hundred-chunk uploads.
                try:
                    self.vector_store.add_documents(documents)
                    logger.info("Added %d chunks to vector store", len(documents))
                except (ValueError, RuntimeError) as batch_error:
                    # Fall back to per-chunk so one bad chunk can't sink the
                    # entire upload. This path is rare with local embeddings
                    # but matters once API-based embeddings (rate limits) are
                    # an option.
                    logger.warning(
                        "Batched add failed (%s); falling back to per-chunk", batch_error
                    )
                    success_count = 0
                    for doc in documents:
                        try:
                            self.vector_store.add_documents([doc])
                            success_count += 1
                        except (ValueError, RuntimeError) as e:
                            logger.warning("Failed to add chunk: %s", e)
                            continue
                    if success_count == 0:
                        logger.error("Failed to add any chunks")
                        return False
                    logger.info(
                        "Added %d/%d chunks via per-chunk fallback", success_count, len(documents)
                    )

            self.save_to_disk()
            return True

        except (ValueError, RuntimeError) as e:
            logger.error("Error adding documents to vector store: %s", e)
            return False

    def soft_delete_document(self, document_id: int):
        """Mark a document as deleted (soft delete)."""
        self.deleted_document_ids.add(document_id)
        self._save_deleted_ids()
        logger.info("Soft-deleted document ID: %d", document_id)

    def undelete_document(self, document_id: int):
        """Restore a soft-deleted document."""
        if document_id in self.deleted_document_ids:
            self.deleted_document_ids.remove(document_id)
            self._save_deleted_ids()
            logger.info("Restored document ID: %d", document_id)

    def is_deleted(self, document_id: int) -> bool:
        """Check if a document is soft-deleted."""
        return document_id in self.deleted_document_ids

    def _build_filter_function(
        self,
        document_id: Optional[int],
        max_chapter: Optional[int],
        include_reference: bool,
    ) -> Callable[[Dict[str, Any]], bool]:
        """Construct the metadata filter for retrieval (spoiler + soft-delete)."""

        def filter_function(metadata: Dict[str, Any]) -> bool:
            doc_id = metadata.get("document_id")

            # 1. Always filter out soft-deleted documents.
            if doc_id in self.deleted_document_ids:
                return False

            # 2. Filter to specific document if requested.
            if document_id is not None and doc_id != document_id:
                return False

            # 3. Spoiler protection (only active when max_chapter is set).
            if max_chapter is not None:
                is_ref = metadata.get("is_reference", False)
                ch_num = metadata.get("chapter_number")

                # Reference material (glossary/appendix) when explicitly opted in.
                if include_reference and is_ref:
                    return True

                # Chunk has a chapter number and is at/before the cutoff.
                if ch_num is not None and ch_num <= max_chapter:
                    return True

                # Chunks without a chapter number (e.g. frontmatter) are blocked
                # under spoiler protection — we cannot prove they are safe.
                return False

            return True

        return filter_function

    def _fetch_k_for(
        self,
        k: int,
        document_id: Optional[int],
        max_chapter: Optional[int],
    ) -> int:
        """Pick how many candidates to pull from FAISS before filtering."""
        if max_chapter is not None:
            return k * 50  # spoiler filter is aggressive; fetch wide
        if document_id is not None:
            return k * 4
        return k * 2

    def get_retriever(
        self,
        k: int = 8,
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None,
        include_reference: bool = False
    ):
        """
        Get a retriever with filtering support.

        SIMPLIFIED SPOILER MODEL:
        - max_chapter=None: Return all content (no spoiler filter)
        - max_chapter=10: Return only chunks where chapter_number <= 10
        - include_reference=True: Also include chunks where is_reference=True
        """
        if self.vector_store is None:
            return None

        search_kwargs = {
            "k": k,
            "fetch_k": self._fetch_k_for(k, document_id, max_chapter),
            "filter": self._build_filter_function(document_id, max_chapter, include_reference),
        }

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def search_with_scores(
        self,
        query: str,
        k: int = 8,
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None,
        include_reference: bool = False,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents along with their FAISS similarity scores,
        applying the same spoiler/soft-delete filtering as get_retriever().

        When a reranker is configured we first pull a wider candidate pool
        (RERANK_POOL_SIZE) and let the cross-encoder reorder it; otherwise
        we just take the top-k directly from FAISS.

        Returns a list of (Document, score) tuples. The score is always the
        original FAISS L2 distance (lower = closer); reranking only changes
        order, never the score values, so normalize_score() math is preserved.
        """
        if self.vector_store is None:
            return []

        filter_fn = self._build_filter_function(document_id, max_chapter, include_reference)

        retrieve_k = max(k, settings.RERANK_POOL_SIZE) if self.reranker else k
        fetch_k = self._fetch_k_for(retrieve_k, document_id, max_chapter)

        candidates = self.vector_store.similarity_search_with_score(
            query,
            k=retrieve_k,
            fetch_k=fetch_k,
            filter=filter_fn,
        )

        if self.reranker is None or len(candidates) <= 1:
            return candidates[:k]

        return self.reranker.rerank(query, candidates, top_k=k)

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """
        Convert FAISS L2 distance to a 0..1 similarity-style score.

        FAISS with HuggingFaceEmbeddings returns squared L2 distance.
        For unit-normalized embeddings (which sentence-transformers produces),
        ||a - b||^2 = 2 - 2*cos(a, b), so cos = 1 - dist/2.
        Clamp to [0, 1] to handle small numerical drift.
        """
        cosine = 1.0 - (raw_score / 2.0)
        return max(0.0, min(1.0, cosine))

    def rebuild_index(self, all_documents: List[Document]) -> bool:
        """Rebuild the vector store from scratch."""
        logger.info("Rebuilding vector store index...")

        try:
            active_docs = [
                doc for doc in all_documents
                if doc.metadata.get("document_id") not in self.deleted_document_ids
            ]

            if not active_docs:
                logger.warning("No active documents to rebuild index")
                self.vector_store = None
                return True

            self.vector_store = FAISS.from_documents(active_docs, self.embeddings)

            old_deleted_count = len(self.deleted_document_ids)
            self.deleted_document_ids.clear()

            self.save_to_disk()

            chunk_count = self.vector_store.index.ntotal
            logger.info(
                "Index rebuilt: %d chunks (physically removed %d soft-deleted documents)",
                chunk_count, old_deleted_count,
            )

            return True

        except (ValueError, RuntimeError, OSError):
            logger.exception("Failed to rebuild index")
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
        except AttributeError:
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
            except AttributeError:
                pass

        return stats

    def clear_all(self):
        """Clear the entire vector store and all tracking."""
        import shutil

        self.vector_store = None
        self.deleted_document_ids.clear()

        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
            logger.info("Cleared all vector store data")