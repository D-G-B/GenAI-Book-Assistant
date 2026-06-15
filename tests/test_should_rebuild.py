"""Tests for VectorStoreManager.should_rebuild.

The rebuild trigger fires when soft-deleted *chunks* exceed a fraction of total
chunks. `deleted_document_ids` tracks documents, not chunks, so the count is
derived from the docstore (one document maps to many chunks). Built via __new__
to skip the heavy __init__ (embeddings); a fake vector_store supplies
index.ntotal + docstore._dict.
"""

import types

from langchain.schema import Document

from app.services.vector_store_manager import VectorStoreManager


def make_vsm(chunk_doc_ids, deleted_ids):
    """VSM whose index holds one chunk per entry in chunk_doc_ids (a document_id)."""
    docstore = types.SimpleNamespace(_dict={
        str(i): Document(page_content="x", metadata={"document_id": d})
        for i, d in enumerate(chunk_doc_ids)
    })
    vs = types.SimpleNamespace(
        index=types.SimpleNamespace(ntotal=len(chunk_doc_ids)),
        docstore=docstore,
    )
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm.vector_store = vs
    vsm.deleted_document_ids = set(deleted_ids)
    return vsm


def test_no_vector_store_returns_false():
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm.vector_store = None
    vsm.deleted_document_ids = {1}
    assert vsm.should_rebuild() is False


def test_nothing_deleted_returns_false():
    assert make_vsm([1, 1, 2, 2], deleted_ids=set()).should_rebuild() is False


def test_deleted_chunk_ratio_above_threshold_triggers():
    # doc 2 owns 2 of 6 chunks -> 0.33 > 0.2
    assert make_vsm([1, 1, 1, 1, 2, 2], deleted_ids={2}).should_rebuild() is True


def test_deleted_chunk_ratio_below_threshold_does_not_trigger():
    # doc 2 owns 1 of 10 chunks -> 0.1 < 0.2
    assert make_vsm([1] * 9 + [2], deleted_ids={2}).should_rebuild() is False


def test_counts_chunks_not_documents():
    # 3 deleted documents, but each owns only 1 of 100 chunks -> 0.03 < 0.2.
    # The old formula deleted_docs/(deleted_docs+10) = 3/13 = 0.23 wrongly fired.
    chunk_ids = [1, 2, 3] + [0] * 97
    assert make_vsm(chunk_ids, deleted_ids={1, 2, 3}).should_rebuild() is False


def test_threshold_is_configurable():
    vsm = make_vsm([1, 1, 1, 1, 2, 2], deleted_ids={2})  # ratio 0.33
    assert vsm.should_rebuild(threshold=0.5) is False
    assert vsm.should_rebuild(threshold=0.3) is True
