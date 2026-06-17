"""Deterministic retrieval-recall *sanity* on the synthetic corpus.

This guards that basic retrieval works (a clearly-answerable query surfaces its
answer chunk in top-k). It is NOT the realistic Baron ranking miss — that is a
big-haystack effect that does not reliably reproduce on a ~12-chunk corpus, so it
lives in the real-Dune harness (`tests/eval/run_eval.py`) instead, where 1573
chunks actually reproduce it. See the plan / current_state.md.

Marked `retrieval`; skip with `-m "not retrieval"`.
"""

import pytest

from tests.fixtures.retrieval_corpus import (
    QUERY_SALT_GATE,
    QUERY_VETHRAN_PLAN,
    SALT_GATE_MARKER,
    USER_A,
)

pytestmark = pytest.mark.retrieval


def contents(results):
    return " ".join(d.page_content for d, _ in results)


def chapters(results):
    return [d.metadata.get("chapter_number") for d, _ in results]


def test_clearly_retrievable_answer_in_topk(retrieval_index):
    """A query with an obvious lexical+semantic match must retrieve its answer chunk."""
    res = retrieval_index.search_with_scores(QUERY_SALT_GATE, k=5, user_id=USER_A)
    assert SALT_GATE_MARKER in contents(res)


def test_plan_query_surfaces_scheming_chapters(retrieval_index):
    """The 'what is he planning' query is dominated by the alive/scheming chapters.

    Mirrors the real Baron case: 'antagonist active' content is highly retrievable.
    """
    res = retrieval_index.search_with_scores(QUERY_VETHRAN_PLAN, k=5, user_id=USER_A)
    assert any(c in (2, 3, 5, 6) for c in chapters(res))


# ---------- offline faithfulness / grounding ----------

def retrieved_supports(results, answer_keywords):
    """Grounding check: do all required answer keywords appear in the retrieved context?

    The offline form of "can the system even answer this given what it retrieved?" —
    if this is False, the LLM cannot answer faithfully no matter how good the prompt.
    """
    ctx = " ".join(d.page_content.lower() for d, _ in results)
    return all(kw.lower() in ctx for kw in answer_keywords)


def test_grounding_present_for_answerable_query(retrieval_index):
    res = retrieval_index.search_with_scores(QUERY_SALT_GATE, k=5, user_id=USER_A)
    assert retrieved_supports(res, ["Provisional Accord", "Salt Gate"])
