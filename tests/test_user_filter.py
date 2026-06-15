"""Tests for the multi-tenancy branch of VectorStoreManager._build_filter_function.

Mirrors test_spoiler_filter.py: the manager is built via __new__ to skip the
HuggingFace embeddings download; the filter only reads self.deleted_document_ids.
The tenant check rejects any chunk whose metadata user_id != the requesting user.
"""

from app.services.vector_store_manager import VectorStoreManager


def make_manager(deleted_ids=None):
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm.deleted_document_ids = set(deleted_ids or [])
    return vsm


def chunk(document_id=1, user_id=1, chapter_number=1, is_reference=False):
    return {
        "document_id": document_id,
        "user_id": user_id,
        "chapter_number": chapter_number,
        "is_reference": is_reference,
    }


def test_filter_rejects_foreign_user():
    vsm = make_manager()
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=None, include_reference=False, user_id=1
    )
    assert fn(chunk(user_id=2)) is False


def test_filter_allows_own_user():
    vsm = make_manager()
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=None, include_reference=False, user_id=1
    )
    assert fn(chunk(user_id=1)) is True


def test_filter_blocks_chunk_without_user_id_when_user_set():
    """Unstamped chunks (no user_id key, e.g. pre-migration) are blocked under a user scope."""
    vsm = make_manager()
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=None, include_reference=False, user_id=1
    )
    assert fn({"document_id": 1, "chapter_number": 1}) is False


def test_no_user_scope_ignores_user_id():
    """user_id=None (internal / eval callers) disables tenant filtering entirely."""
    vsm = make_manager()
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=None, include_reference=False
    )
    assert fn(chunk(user_id=999)) is True


def test_user_filter_composes_with_spoiler():
    """A foreign user's chunk is blocked even when it's within the chapter window."""
    vsm = make_manager()
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=10, include_reference=False, user_id=1
    )
    assert fn(chunk(user_id=2, chapter_number=3)) is False
    assert fn(chunk(user_id=1, chapter_number=3)) is True


def test_soft_delete_precedes_user_check():
    """Soft-deleted docs are blocked first, even for the owning user."""
    vsm = make_manager(deleted_ids={5})
    fn = vsm._build_filter_function(
        document_id=None, max_chapter=None, include_reference=False, user_id=1
    )
    assert fn(chunk(document_id=5, user_id=1)) is False
