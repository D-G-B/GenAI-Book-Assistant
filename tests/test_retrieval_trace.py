"""Tests for the DEBUG retrieval-trace in VectorStoreManager.search_with_scores.

The trace is the diagnostic that would have made the Baron miss obvious: it shows
which chunks were retrieved (chapter / is_reference / score), the filter survivor
count, and any rerank reorder. Two index-backed tests assert its content (marked
`retrieval`); one fast test asserts the perf contract (silent at INFO, skipped when
DEBUG is off) without loading the embedding model.
"""

import logging
import types

import pytest

from tests.fixtures.retrieval_corpus import (
    DEATH_CHAPTER,
    QUERY_THRONE,
    QUERY_VETHRAN_PLAN,
    USER_A,
)

TRACE_LOGGER = "app.services.vector_store_manager.trace"


@pytest.mark.retrieval
def test_trace_logs_retrieved_chunks_at_debug(retrieval_index, caplog):
    with caplog.at_level(logging.DEBUG, logger=TRACE_LOGGER):
        retrieval_index.search_with_scores(
            QUERY_THRONE, k=5, user_id=USER_A, max_chapter=DEATH_CHAPTER
        )
    text = caplog.text
    assert "retrieval trace" in text
    assert "top-k returned" in text
    assert "reference_in_topk=" in text
    # the throne query surfaces the death chapter; the trace names its chapter.
    assert f"ch={DEATH_CHAPTER}" in text


class _FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))[:top_k]


@pytest.mark.retrieval
def test_trace_shows_rerank_reorder(retrieval_index, caplog, monkeypatch):
    monkeypatch.setattr(retrieval_index, "reranker", _FakeReranker())
    with caplog.at_level(logging.DEBUG, logger=TRACE_LOGGER):
        retrieval_index.search_with_scores(QUERY_VETHRAN_PLAN, k=5, user_id=USER_A)
    assert "rerank order" in caplog.text


def test_trace_silent_at_info_and_skipped_when_debug_off(caplog):
    """Perf contract: nothing logs at INFO; the DEBUG trace work is skipped. No model load."""
    from langchain.schema import Document

    from app.services.vector_store_manager import VectorStoreManager

    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm.reranker = None
    vsm.deleted_document_ids = set()
    doc = Document(
        page_content="x",
        metadata={"document_id": 1, "chapter_number": 1, "is_reference": False, "document_title": "t"},
    )
    vsm.vector_store = types.SimpleNamespace(
        similarity_search_with_score=lambda query, k, fetch_k, filter: [(doc, 0.5)]
    )

    with caplog.at_level(logging.INFO):
        results = vsm.search_with_scores("q", k=3)

    assert results  # still returns
    assert "retrieval trace" not in caplog.text  # trace is DEBUG-only
