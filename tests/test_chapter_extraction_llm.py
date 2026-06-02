"""Tests for the experimental LLM chapter detector and the chapter metric.

Deterministic: the LLM is injected via the `invoke` seam (a plain function
returning canned JSON), so these run in the fast suite with no API keys.
"""

import json

from app.services.llm_chapter_detector import (
    detect_chapters_llm,
    heading_candidates,
    _parse_llm_json_array,
)
from tests.eval.chapter_metrics import score_detection, resolve_offsets


def fake_invoke(text):
    """Build an injectable invoke() that returns a fixed LLM response body."""
    def _invoke(_prompt):
        return {"text": text, "provider": "fake", "calls": 1}
    return _invoke


# ---------- heading_candidates ----------

def test_heading_candidates_finds_standalone_short_lines():
    content = "The Gathering Storm\nbody text here.\n\nA Meeting at Dusk\nmore body."
    cands = heading_candidates(content)
    titles = [t for _, t in cands]
    assert "The Gathering Storm" in titles
    assert "A Meeting at Dusk" in titles
    # Offsets must point at the real heading positions.
    by_title = {t: off for off, t in cands}
    assert by_title["The Gathering Storm"] == content.index("The Gathering Storm")
    assert by_title["A Meeting at Dusk"] == content.index("A Meeting at Dusk")


def test_heading_candidates_skips_sentence_lines():
    content = "This is a long ordinary sentence that ends with a period.\n"
    assert heading_candidates(content) == []


# ---------- _parse_llm_json_array ----------

def test_parse_strips_markdown_fences():
    text = '```json\n[{"id": 1, "title": "X"}]\n```'
    assert _parse_llm_json_array(text) == [{"id": 1, "title": "X"}]


def test_parse_returns_none_on_garbage():
    assert _parse_llm_json_array("not json at all") is None
    assert _parse_llm_json_array("") is None


# ---------- detect_chapters_llm ----------

def test_reconciles_ids_to_offsets():
    content = "The Gathering Storm\nbody...\n\nA Meeting at Dusk\nmore body..."
    resp = json.dumps([
        {"id": 1, "title": "The Gathering Storm", "chapter_number": 1, "is_reference": False},
        {"id": 2, "title": "A Meeting at Dusk", "chapter_number": 2, "is_reference": False},
    ])
    out = detect_chapters_llm(content, invoke=fake_invoke(resp))
    assert [c["chapter_number"] for c in out] == [1, 2]
    assert out[0]["start"] == content.index("The Gathering Storm")
    assert out[1]["start"] == content.index("A Meeting at Dusk")


def test_preserves_is_reference_for_afterword():
    content = "Embers\nbody...\n\nAfterword\nnote..."
    resp = json.dumps([
        {"id": 1, "title": "Embers", "chapter_number": 1, "is_reference": False},
        {"id": 2, "title": "Afterword", "chapter_number": None, "is_reference": True},
    ])
    out = detect_chapters_llm(content, invoke=fake_invoke(resp))
    assert out[1]["is_reference"] is True
    assert out[1]["chapter_number"] is None


def test_malformed_json_returns_empty():
    content = "Embers\nbody...\n\nAshfall\nmore..."
    out = detect_chapters_llm(content, invoke=fake_invoke("totally broken"))
    assert out == []


def test_invoke_raising_returns_empty():
    content = "Embers\nbody...\n\nAshfall\nmore..."

    def boom(_prompt):
        raise RuntimeError("provider down")

    assert detect_chapters_llm(content, invoke=boom) == []


def test_unlocatable_title_is_dropped_others_kept():
    content = "Embers\nbody...\n\nAshfall\nmore..."
    resp = json.dumps([
        {"id": 1, "title": "Embers", "chapter_number": 1, "is_reference": False},
        # bad id + a title that doesn't appear in the text → dropped
        {"id": 99, "title": "Nonexistent Heading", "chapter_number": 2, "is_reference": False},
    ])
    out = detect_chapters_llm(content, invoke=fake_invoke(resp))
    assert [c["title"] for c in out] == ["Embers"]


def test_no_candidates_returns_empty_without_calling_llm():
    called = []

    def tracking(_prompt):
        called.append(1)
        return {"text": "[]"}

    # No standalone short heading lines at all.
    content = "This is just one long flowing sentence with no headings to speak of."
    assert detect_chapters_llm(content, invoke=tracking) == []
    assert called == []  # short-circuited before the LLM call


# ---------- chapter_metrics ----------

def test_metric_perfect_match():
    text = "The Gathering Storm\nbody.\n\nA Meeting at Dusk\nmore."
    expected = [
        {"title": "The Gathering Storm", "chapter_number": 1, "is_reference": False},
        {"title": "A Meeting at Dusk", "chapter_number": 2, "is_reference": False},
    ]
    predicted = resolve_offsets(expected, text)
    s = score_detection(predicted, expected, text)
    assert s["boundary_f1"] == 1.0
    assert s["chapter_number_accuracy"] == 1.0
    assert s["is_reference_accuracy"] == 1.0


def test_metric_partial_recall():
    text = "The Gathering Storm\nbody.\n\nA Meeting at Dusk\nmore."
    expected = [
        {"title": "The Gathering Storm", "chapter_number": 1, "is_reference": False},
        {"title": "A Meeting at Dusk", "chapter_number": 2, "is_reference": False},
    ]
    # Detector found only the first heading.
    predicted = [{"start": text.index("The Gathering Storm"), "title": "The Gathering Storm",
                  "chapter_number": 1, "is_reference": False}]
    s = score_detection(predicted, expected, text)
    assert s["boundary_recall"] == 0.5
    assert s["boundary_precision"] == 1.0


def test_metric_empty_prediction_scores_zero():
    text = "The Gathering Storm\nbody."
    expected = [{"title": "The Gathering Storm", "chapter_number": 1, "is_reference": False}]
    s = score_detection([], expected, text)
    assert s["boundary_f1"] == 0.0
    assert s["boundary_recall"] == 0.0
