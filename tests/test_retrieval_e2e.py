"""End-to-end retrieval tests through a REAL FAISS index (not the filter closure).

These build a tiny real index from the synthetic Baron-mode corpus and call
`VectorStoreManager.search_with_scores`, so they cover what the closure-only tests
(test_spoiler_filter.py / test_user_filter.py) cannot: that filtering composes with
actual ranked retrieval. Marked `retrieval` (loads the embedding model) — skip with
`-m "not retrieval"`.

The spoiler / reference / tenancy / soft-delete assertions are deterministic because
they test FILTERING (a chunk that's filtered can never be returned), using queries
that strongly match the target chunk so it WOULD surface if allowed.
"""

import pytest

from tests.fixtures.retrieval_corpus import (
    DEATH_CHAPTER,
    DEATH_MARKER,
    DOC_DELETABLE,
    DOC_MAIN,
    DOC_OTHER_USER,
    OTHER_USER_MARKER,
    QUERY_NOBLE_HOUSES,
    QUERY_SALT_GATE,
    QUERY_THRONE,
    QUERY_VETHRAN_PLAN,
    REFERENCE_MARKER,
    SALT_GATE_MARKER,
    USER_A,
    USER_B,
)

pytestmark = pytest.mark.retrieval


def chapters(results):
    return [d.metadata.get("chapter_number") for d, _ in results]


def contents(results):
    return " ".join(d.page_content for d, _ in results)


# ---------- spoiler ----------

def test_spoiler_blocks_late_chapter(retrieval_index):
    vsm = retrieval_index
    blocked = vsm.search_with_scores(
        QUERY_THRONE, k=8, user_id=USER_A, max_chapter=DEATH_CHAPTER - 1
    )
    assert all(c is not None and c <= DEATH_CHAPTER - 1 for c in chapters(blocked))
    assert DEATH_MARKER not in contents(blocked)

    allowed = vsm.search_with_scores(
        QUERY_THRONE, k=8, user_id=USER_A, max_chapter=DEATH_CHAPTER
    )
    assert DEATH_MARKER in contents(allowed)  # in-range now → retrievable


# ---------- reference inclusion/exclusion (the exact Baron mode) ----------

def test_reference_excluded_by_default_then_included(retrieval_index):
    vsm = retrieval_index
    no_ref = vsm.search_with_scores(
        QUERY_NOBLE_HOUSES, k=8, user_id=USER_A,
        max_chapter=DEATH_CHAPTER, include_reference=False,
    )
    assert REFERENCE_MARKER not in contents(no_ref)

    with_ref = vsm.search_with_scores(
        QUERY_NOBLE_HOUSES, k=8, user_id=USER_A,
        max_chapter=DEATH_CHAPTER, include_reference=True,
    )
    assert REFERENCE_MARKER in contents(with_ref)


# ---------- tenancy ----------

def test_tenancy_isolates_users(retrieval_index):
    vsm = retrieval_index
    res_a = vsm.search_with_scores(QUERY_VETHRAN_PLAN, k=8, user_id=USER_A)
    assert all(d.metadata["document_id"] != DOC_OTHER_USER for d, _ in res_a)
    assert OTHER_USER_MARKER not in contents(res_a)

    res_b = vsm.search_with_scores(QUERY_VETHRAN_PLAN, k=8, user_id=USER_B)
    assert res_b  # USER_B owns one matching chunk
    assert all(d.metadata["document_id"] == DOC_OTHER_USER for d, _ in res_b)


# ---------- soft delete ----------

def test_soft_delete_excludes_then_restores(retrieval_index):
    vsm = retrieval_index
    before = vsm.search_with_scores(QUERY_SALT_GATE, k=8, user_id=USER_A)
    assert SALT_GATE_MARKER in contents(before)

    vsm.soft_delete_document(DOC_DELETABLE)
    try:
        after = vsm.search_with_scores(QUERY_SALT_GATE, k=8, user_id=USER_A)
        assert SALT_GATE_MARKER not in contents(after)
    finally:
        vsm.deleted_document_ids.discard(DOC_DELETABLE)  # restore the shared index


# ---------- k after aggressive fetch under spoiler ----------

def test_k_respected_under_spoiler_fetch(retrieval_index):
    vsm = retrieval_index
    res = vsm.search_with_scores(
        QUERY_VETHRAN_PLAN, k=3, user_id=USER_A, max_chapter=DEATH_CHAPTER
    )
    assert 1 <= len(res) <= 3
    for d, _ in res:
        assert d.metadata["chapter_number"] is not None
        assert d.metadata["chapter_number"] <= DEATH_CHAPTER
        assert d.metadata["document_id"] == DOC_MAIN
        assert d.metadata["user_id"] == USER_A
