"""Pytest configuration for the test suite.

The legacy `test_conversational.py` and `test_reading_partner.py` are manual
scripts that POST to a live server on localhost:8000. Skip them under pytest;
they should be invoked directly with `uv run python tests/test_*.py`.
"""

collect_ignore_glob = [
    "test_conversational.py",
    "test_reading_partner.py",
    "add_complex_document.py",
]

import os

import pytest


@pytest.fixture(scope="session")
def retrieval_index(tmp_path_factory):
    """Build ONE real FAISS index from the synthetic corpus (local MiniLM embeddings).

    Session-scoped so the embedding model loads once. Persists to a throwaway tmp dir
    (never touches ./faiss_index) and pins the reranker OFF so search is reproducible.

    `retrieval`-marked tests depend on this; if the embedding model can't load offline
    the whole layer skips (keeps a cache-cold CI green) — set RAG_REQUIRE_RETRIEVAL_TESTS=1
    to turn that skip into a hard failure (e.g. a CI step that pre-warms the model).
    """
    from app.config import settings
    from tests.fixtures.retrieval_corpus import build_documents

    persist = tmp_path_factory.mktemp("faiss_retrieval")
    prev_reranker = settings.RERANKER_ENABLED
    settings.RERANKER_ENABLED = False
    try:
        from app.services.vector_store_manager import VectorStoreManager

        try:
            vsm = VectorStoreManager(persist_path=str(persist))
        except Exception as exc:  # noqa: BLE001 - model download/load can fail offline
            if os.getenv("RAG_REQUIRE_RETRIEVAL_TESTS") == "1":
                raise
            pytest.skip(f"retrieval tests need the embedding model: {exc}")

        if not vsm.add_documents(build_documents()):
            pytest.skip("failed to build the synthetic retrieval index")

        yield vsm
    finally:
        settings.RERANKER_ENABLED = prev_reranker
