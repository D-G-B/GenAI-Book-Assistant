"""Tests for DocumentManager._detect_chapters_in_content.

Constructs the manager via __new__ to avoid the heavy __init__ that wires the
vector store. The detector only uses self._roman_to_int (a pure helper).
"""

import pytest

from app.services.document_manager import DocumentManager


def make_manager():
    return DocumentManager.__new__(DocumentManager)


# === Body chapters: numbered ===

def test_arabic_numbered_chapters():
    dm = make_manager()
    content = "Chapter 1\nIntro text\n\nChapter 2\nMore text"
    chapters = dm._detect_chapters_in_content(content)
    numbers = [c["chapter_number"] for c in chapters]
    assert numbers == [1, 2]
    assert all(c["is_reference"] is False for c in chapters)


def test_word_numbered_chapters():
    dm = make_manager()
    # Note: chapter markers must be >20 chars apart or the detector dedupes them.
    content = (
        "Chapter One\nThe story begins with a long opening passage.\n\n"
        "Chapter Three\nThe story continues."
    )
    chapters = dm._detect_chapters_in_content(content)
    numbers = [c["chapter_number"] for c in chapters]
    assert numbers == [1, 3]


def test_roman_numbered_chapters():
    dm = make_manager()
    content = (
        "Chapter I\nThe story begins with a long opening passage.\n\n"
        "Chapter IV\nThe story continues."
    )
    chapters = dm._detect_chapters_in_content(content)
    numbers = [c["chapter_number"] for c in chapters]
    assert numbers == [1, 4]


def test_uppercase_chapter_marker():
    dm = make_manager()
    content = "CHAPTER 5\nText"
    chapters = dm._detect_chapters_in_content(content)
    assert len(chapters) == 1
    assert chapters[0]["chapter_number"] == 5


# === Reference sections ===

@pytest.mark.parametrize(
    "marker",
    ["Glossary", "Appendix", "Afterword", "Bibliography", "GLOSSARY", "APPENDIX A"],
)
def test_reference_markers_classified_as_reference(marker):
    dm = make_manager()
    content = f"Chapter 1\nStory begins\n\n{marker}\nReference content here"
    chapters = dm._detect_chapters_in_content(content)
    refs = [c for c in chapters if c["is_reference"]]
    assert len(refs) == 1
    assert refs[0]["chapter_number"] is None


def test_chapter_and_reference_coexist_in_order():
    dm = make_manager()
    content = (
        "Chapter 1\nStory begins\n\n"
        "Chapter 2\nMore story\n\n"
        "Glossary\nTerm definitions"
    )
    chapters = dm._detect_chapters_in_content(content)
    assert len(chapters) == 3
    assert chapters[0]["chapter_number"] == 1 and chapters[0]["is_reference"] is False
    assert chapters[1]["chapter_number"] == 2 and chapters[1]["is_reference"] is False
    assert chapters[2]["chapter_number"] is None and chapters[2]["is_reference"] is True


# === Edge cases ===

def test_no_chapter_markers_returns_empty():
    dm = make_manager()
    content = "Just some prose with no markers anywhere."
    assert dm._detect_chapters_in_content(content) == []


def test_chapters_sorted_by_position_not_discovery_order():
    """Reference patterns are matched after chapter patterns, but the final list
    must be sorted by position so chunking walks the content in order."""
    dm = make_manager()
    content = (
        "Glossary\nReference up top.\n\n"
        "Chapter 1\nStory begins.\n\n"
        "Appendix A\nMore reference."
    )
    chapters = dm._detect_chapters_in_content(content)
    starts = [c["start"] for c in chapters]
    assert starts == sorted(starts)
