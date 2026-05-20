"""Tests for VectorStoreManager._build_filter_function — the spoiler-filter IP.

Constructs the manager via __new__ to avoid the HuggingFace embeddings download
in __init__. The filter only reads self.deleted_document_ids.
"""

import pytest

from app.services.vector_store_manager import VectorStoreManager


def make_manager(deleted_ids=None):
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm.deleted_document_ids = set(deleted_ids or [])
    return vsm


def chunk(document_id=1, chapter_number=None, is_reference=False):
    return {
        "document_id": document_id,
        "chapter_number": chapter_number,
        "is_reference": is_reference,
    }


# === No spoiler filter (max_chapter=None) ===

@pytest.mark.parametrize(
    "metadata",
    [
        chunk(chapter_number=1),
        chunk(chapter_number=99),
        chunk(chapter_number=None),
        chunk(chapter_number=None, is_reference=True),
    ],
)
def test_no_spoiler_filter_passes_all(metadata):
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=None, include_reference=False)
    assert fn(metadata) is True


# === Spoiler filter active ===

def test_spoiler_blocks_future_chapter():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=False)
    assert fn(chunk(chapter_number=11)) is False


def test_spoiler_allows_current_chapter():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=False)
    assert fn(chunk(chapter_number=10)) is True


def test_spoiler_allows_past_chapter():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=False)
    assert fn(chunk(chapter_number=1)) is True


def test_spoiler_blocks_chunk_with_no_chapter_number():
    """Frontmatter / unknown-chapter chunks must be blocked under spoiler protection.

    We cannot prove a chapter-less chunk is safe, so the filter is conservative.
    """
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=False)
    assert fn(chunk(chapter_number=None)) is False


# === Reference toggle ===

def test_reference_blocked_when_include_reference_false():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=False)
    assert fn(chunk(chapter_number=None, is_reference=True)) is False


def test_reference_allowed_when_include_reference_true():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=True)
    assert fn(chunk(chapter_number=None, is_reference=True)) is True


def test_reference_toggle_irrelevant_without_spoiler():
    """include_reference only matters when max_chapter is set."""
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=None, max_chapter=None, include_reference=False)
    assert fn(chunk(chapter_number=None, is_reference=True)) is True


# === Soft delete ===

def test_soft_deleted_always_blocked_no_spoiler():
    vsm = make_manager(deleted_ids={42})
    fn = vsm._build_filter_function(document_id=None, max_chapter=None, include_reference=False)
    assert fn(chunk(document_id=42, chapter_number=1)) is False


def test_soft_deleted_always_blocked_with_spoiler():
    vsm = make_manager(deleted_ids={42})
    fn = vsm._build_filter_function(document_id=None, max_chapter=10, include_reference=True)
    assert fn(chunk(document_id=42, chapter_number=1, is_reference=True)) is False


def test_non_deleted_doc_passes_when_others_deleted():
    vsm = make_manager(deleted_ids={42})
    fn = vsm._build_filter_function(document_id=None, max_chapter=None, include_reference=False)
    assert fn(chunk(document_id=99, chapter_number=1)) is True


# === Document ID filter ===

def test_document_id_filter_blocks_other_docs():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=1, max_chapter=None, include_reference=False)
    assert fn(chunk(document_id=2, chapter_number=1)) is False


def test_document_id_filter_passes_target_doc():
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=1, max_chapter=None, include_reference=False)
    assert fn(chunk(document_id=1, chapter_number=1)) is True


# === Combined: spoiler + reference + soft-delete + document filter ===

@pytest.mark.parametrize(
    "doc_id,chapter,is_ref,expected",
    [
        # Target doc, body chapter at cutoff → pass
        (1, 5, False, True),
        # Target doc, body chapter past cutoff → block
        (1, 6, False, False),
        # Target doc, reference with include_reference=True → pass
        (1, None, True, True),
        # Target doc, reference but no chapter and not is_reference under spoiler → block
        (1, None, False, False),
        # Wrong doc → block regardless of chapter
        (2, 1, False, False),
        # Wrong doc, even reference → block
        (2, None, True, False),
    ],
)
def test_combined_filters(doc_id, chapter, is_ref, expected):
    vsm = make_manager()
    fn = vsm._build_filter_function(document_id=1, max_chapter=5, include_reference=True)
    assert fn(chunk(document_id=doc_id, chapter_number=chapter, is_reference=is_ref)) is expected
