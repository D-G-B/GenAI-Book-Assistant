"""Cross-encoder reranker for retrieval results.

Wraps a sentence-transformers CrossEncoder. Lazy-loads the model on first
rerank() call so module import is cheap and pytest collection is unaffected.

The reranker reorders the candidate pool from FAISS but preserves the
original L2 score in each returned tuple — the score's cosine semantics
(via VectorStoreManager.normalize_score) are unchanged for API consumers.
The win is the new ordering, not a new score scale.
"""

import logging
from typing import List, Tuple

from langchain.schema import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Reorder candidates by cross-encoder relevance and trim to top_k.

        The returned tuples keep their original FAISS L2 scores so downstream
        normalize_score() math is unaffected.
        """
        if not candidates:
            return []
        self._ensure_model()

        pairs = [(query, doc.page_content) for doc, _ in candidates]
        rerank_scores = self._model.predict(pairs)

        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [orig for orig, _ in scored[:top_k]]
