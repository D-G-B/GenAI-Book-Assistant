"""Tests for wiring the hybrid LLM chapter detector into ingest.

Covers DocumentManager._process_and_chunk's detector selection: prefer the
hybrid detector when it labels >= 2 sections, otherwise fall back to the regex
detector. The LLM is injected via the `invoke` seam (canned JSON), so these run
in the fast suite with no API keys and no network.

The manager is built via __new__ to skip the heavy __init__ (vector store,
embeddings); we attach only the text_splitter the chunking path needs.
"""

import functools
import json
import types

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.document_manager import DocumentManager


# A marked book where the hybrid and regex detectors disagree, so a test can
# tell which path drove chunking:
#   regex  -> Section 1/2/3 numbered 1/2/3 (file-section numbers), Glossary NOT
#             flagged reference, ToC "Chapter One" line counted.
#   hybrid -> Section 1 (ToC) omitted, Section 2/3 renumbered to story order
#             1/2, Glossary flagged is_reference.
MARKED_BOOK = (
    "=== Section 1 ===\n\nTable of Contents\n\nChapter One\nChapter Two\n\n"
    "=== Section 2 ===\n\nA quote about beginnings.\n"
    "THE STORY opened on a cold morning in the high desert, wind tugging at the cloaks.\n\n"
    "=== Section 3 ===\n\nAnother epigraph here.\n"
    "THE STORY continued apace through the long afternoon and into the gathering dark.\n\n"
    "=== Glossary ===\n\nABA: a loose robe worn by the desert folk of the deep south, "
    "dyed in muted ochre and bound at the waist with a woven cord.\n"
)

# What a correct hybrid LLM call returns for MARKED_BOOK: omit the ToC section
# (id 1), number the narrative sections in story order, flag the glossary.
HYBRID_RESPONSE = json.dumps([
    {"id": 2, "title": "=== Section 2 ===", "chapter_number": 1, "is_reference": False},
    {"id": 3, "title": "=== Section 3 ===", "chapter_number": 2, "is_reference": False},
    {"id": 4, "title": "=== Glossary ===", "chapter_number": None, "is_reference": True},
])


def fake_invoke(text):
    """Build an injectable invoke() that returns a fixed LLM response body."""
    def _invoke(_prompt):
        return {"text": text, "provider": "fake", "calls": 1}
    return _invoke


def make_dm():
    dm = DocumentManager.__new__(DocumentManager)
    dm.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return dm


def fake_doc(content):
    return types.SimpleNamespace(id=1, title="Test Book", source_type="epub", content=content)


def chapter_numbers(chunks):
    return {c.metadata["chapter_number"] for c in chunks}


# ---------- helper: _detect_chapters_hybrid ----------

async def test_detect_chapters_hybrid_runs_detector_through_executor():
    """The helper passes the injected invoke to the detector and returns its output."""
    dm = make_dm()
    out = await dm._detect_chapters_hybrid(MARKED_BOOK, invoke=fake_invoke(HYBRID_RESPONSE))
    assert [c["chapter_number"] for c in out] == [1, 2, None]
    assert out[-1]["is_reference"] is True


async def test_detect_chapters_hybrid_returns_empty_on_llm_failure():
    dm = make_dm()
    out = await dm._detect_chapters_hybrid(MARKED_BOOK, invoke=fake_invoke("not json"))
    assert out == []


# ---------- _process_and_chunk: detector selection ----------

async def test_process_and_chunk_uses_hybrid_labels_when_available():
    """When the hybrid detector finds >= 2 sections, chunks carry ITS labels."""
    dm = make_dm()
    dm._detect_chapters_hybrid = functools.partial(
        DocumentManager._detect_chapters_hybrid, dm, invoke=fake_invoke(HYBRID_RESPONSE)
    )

    chunks = await dm._process_and_chunk(fake_doc(MARKED_BOOK))

    body = {c.metadata["chapter_number"] for c in chunks
            if c.metadata["chapter_number"] is not None and not c.metadata["is_reference"]}
    # Hybrid renumbered to story order 1, 2 and never emits the file-section 3.
    assert body == {1, 2}
    assert 3 not in chapter_numbers(chunks)
    # The glossary was flagged reference (regex would leave it False).
    assert any(c.metadata["is_reference"] for c in chunks)


async def test_process_and_chunk_falls_back_to_regex_on_hybrid_failure():
    """When the hybrid detector returns [] (LLM failure), the regex path runs."""
    dm = make_dm()
    dm._detect_chapters_hybrid = functools.partial(
        DocumentManager._detect_chapters_hybrid, dm, invoke=fake_invoke("not json")
    )

    chunks = await dm._process_and_chunk(fake_doc(MARKED_BOOK))

    # Regex uses the file-section numbers, so chapter 3 appears — proof the
    # fallback ran rather than the hybrid labels.
    assert 3 in chapter_numbers(chunks)
    # Regex does not flag the glossary as reference (documented quirk).
    assert not any(c.metadata["is_reference"] for c in chunks)


# A parseable-but-wrong hybrid result: the LLM copied the marker numbers (Section
# 2/3 -> chapter 2/3) instead of renumbering to story order. Complete and parseable,
# so it would pass the len >= 2 gate — but the sanity check rejects it.
HYBRID_MISLABELED = json.dumps([
    {"id": 2, "title": "=== Section 2 ===", "chapter_number": 2, "is_reference": False},
    {"id": 3, "title": "=== Section 3 ===", "chapter_number": 3, "is_reference": False},
    {"id": 4, "title": "=== Glossary ===", "chapter_number": None, "is_reference": True},
])


async def test_process_and_chunk_falls_back_to_regex_on_implausible_hybrid():
    """A complete-but-mislabeled hybrid result falls back to the regex detector."""
    dm = make_dm()
    dm._detect_chapters_hybrid = functools.partial(
        DocumentManager._detect_chapters_hybrid, dm, invoke=fake_invoke(HYBRID_MISLABELED)
    )

    chunks = await dm._process_and_chunk(fake_doc(MARKED_BOOK))

    # The hybrid result was rejected (non-story-order numbering), so regex drove
    # chunking: it uses the file-section numbers, so chapter 3 appears and the
    # glossary is not flagged reference (documented regex quirk).
    assert 3 in chapter_numbers(chunks)
    assert not any(c.metadata["is_reference"] for c in chunks)


async def test_process_and_chunk_skips_hybrid_when_disabled(monkeypatch):
    """LLM_CHAPTER_DETECTION_ENABLED=False: the hybrid detector is never called."""
    monkeypatch.setattr(settings, "LLM_CHAPTER_DETECTION_ENABLED", False)

    dm = make_dm()
    called = []

    async def _should_not_run(*_args, **_kwargs):
        called.append(1)
        return []

    dm._detect_chapters_hybrid = _should_not_run

    chunks = await dm._process_and_chunk(fake_doc(MARKED_BOOK))

    assert called == []  # hybrid path skipped entirely
    assert 3 in chapter_numbers(chunks)  # regex drove chunking
