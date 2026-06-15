"""PageIndex-style LLM chapter detector (experimental, standalone).

This module is an *experiment*: it borrows PageIndex's idea of using an LLM to
recognise a document's chapter structure instead of regex pattern-matching. It
is deliberately NOT wired into the ingestion path — it exists so we can measure
it head-to-head against the existing regex detector
(`DocumentManager._detect_chapters_in_content`) before deciding how, or whether,
to integrate it. See `.claude/plans/` and `current_state.md` (Known issues).

Design notes:
- Returns the *identical* dict shape as the regex detector
  ({"start", "title", "chapter_number", "is_reference"}, sorted by start) so the
  comparison is apples-to-apples and any future wiring is trivial.
- Books can be 100k+ tokens, so we do NOT send the whole document. We pre-filter
  to candidate *heading lines* and send only those. This is a deliberate
  simplification of PageIndex's full token-windowing; adequate to test the idea
  cheaply.
- The LLM client is injectable (`invoke`) purely so tests can run deterministically
  without API keys. Production callers omit it and get the real provider-fallback
  client (`enhanced_rag_service.invoke_with_fallback`).
- Any failure (LLM error, bad JSON, nothing usable) returns [] so callers can
  fall back to the regex result. We never raise.
- Two variants are measured head-to-head by the eval: `detect_chapters_llm`
  (LLM picks boundaries AND labels them, from bare heading lines) and
  `detect_chapters_hybrid` (boundaries come deterministically from the ingest
  extractor's `=== ... ===` markers; the LLM only labels each anchor, seeing a
  short snippet of the text that follows it). The hybrid exists because the
  real-book eval showed boundary selection is what the LLM fails at — its
  labelling of found boundaries was perfect — and anchoring shrinks the prompt
  ~10x.
"""

import json
import logging
import re
from typing import Callable, List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Heading heuristics: a "candidate" line is short, standalone, and not a sentence.
_MAX_HEADING_LEN = 80
_MAX_HEADING_WORDS = 12
_SENTENCE_ENDINGS = (".", ",", ";", ":")

# Section markers stamped by the ingest extractors (one per epub spine item; see
# documents_routes._extract_epub_content). When present they ARE the document's
# section boundaries, so the hybrid detector anchors on them instead of asking
# the LLM to pick boundaries out of heading-shaped noise.
_MARKER_RE = re.compile(r"^===\s*.+?\s*===\s*$", re.MULTILINE)

# How much following body text to show the LLM per anchor (whitespace-collapsed).
_SNIPPET_LEN = 120


def heading_candidates(content: str) -> List[Tuple[int, str]]:
    """Return (char_offset, line_text) for lines that look like headings.

    A line qualifies when it is non-empty, short (few words, under the length
    cap), does not read like a sentence (no trailing sentence punctuation), and
    is standalone (first line, or preceded by a blank line). The absolute char
    offset is preserved so detected headings map back to exact positions in
    `content` without a fragile substring search.
    """
    candidates: List[Tuple[int, str]] = []
    offset = 0
    prev_blank = True  # treat start-of-document as following a blank line
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        is_blank = not stripped

        if (
            not is_blank
            and prev_blank
            and len(stripped) <= _MAX_HEADING_LEN
            and len(stripped.split()) <= _MAX_HEADING_WORDS
            and not stripped.endswith(_SENTENCE_ENDINGS)
        ):
            candidates.append((offset, stripped))

        prev_blank = is_blank
        offset += len(line)

    return candidates


def _parse_llm_json_array(text: str) -> Optional[list]:
    """Extract a JSON array from an LLM response, tolerating markdown fences.

    Returns the parsed list, or None on any failure.
    """
    if not text:
        return None
    # Take everything between the first '[' and the last ']' — this strips both
    # ```json fences and any surrounding prose the model may have added.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _looks_truncated(text: str) -> bool:
    """Heuristic: a response that opened a JSON array but never closed it.

    The labelling output is an array of objects, so the only ']' is the array's
    closing bracket; if the model was cut off at its output-token cap the text
    has a '[' and no ']'. Lets the caller distinguish a truncated response
    (actionable: raise the cap) from other parse failures (empty / no JSON).
    """
    return "[" in text and "]" not in text


def _build_prompt(candidates: List[Tuple[int, str]]) -> str:
    """Build the extraction prompt from numbered candidate heading lines."""
    numbered = "\n".join(f"{i}: {text}" for i, (_, text) in enumerate(candidates, start=1))
    return (
        "You are extracting the chapter structure of a book. Below are candidate "
        "heading lines, each prefixed with a numeric ID. Identify which lines are "
        "real chapter or section starts, and which are reference / back-matter "
        "(glossary, appendix, afterword, epilogue, bibliography, notes, about the "
        "author).\n\n"
        "Return ONLY a JSON array, with no prose and no markdown fences:\n"
        '[{"id": <int>, "title": "<verbatim line>", "chapter_number": <int|null>, '
        '"is_reference": <true|false>}, ...]\n\n'
        "Rules:\n"
        "- chapter_number: sequential story position (1, 2, 3, ...) for narrative "
        "chapters; null for reference material and unnumbered front/back matter.\n"
        "- is_reference: true for glossary/appendix/afterword/epilogue/"
        "bibliography/notes/about-the-author; false otherwise.\n"
        "- Only include IDs that are genuine chapter or reference headings; omit "
        "running headers, page numbers, and body text.\n\n"
        "Candidate lines:\n"
        f"{numbered}\n"
    )


def _default_invoke(prompt: str) -> Dict:
    """Resolve the real LLM client lazily (keeps import-time coupling out)."""
    from app.services.enhanced_rag_service import enhanced_rag_service

    return enhanced_rag_service.invoke_with_fallback(prompt)


def _reconcile(parsed: list, anchors: List[Tuple[int, str]], content: str) -> List[Dict]:
    """Map LLM-returned items back to authoritative anchor offsets.

    Using the id avoids the classic bug where content.find(title) matches a copy
    of the title inside body prose rather than the real heading. Items with no
    locatable position are dropped.
    """
    detected: List[Dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        offset: Optional[int] = None
        title = item.get("title")

        if isinstance(idx, int) and 1 <= idx <= len(anchors):
            offset, title = anchors[idx - 1]
        elif isinstance(title, str):
            found = content.find(title)
            offset = found if found != -1 else None

        if offset is None:
            continue  # not locatable — drop it

        ch_num = item.get("chapter_number")
        detected.append({
            "start": offset,
            "title": title,
            "chapter_number": ch_num if isinstance(ch_num, int) else None,
            "is_reference": bool(item.get("is_reference", False)),
        })

    detected.sort(key=lambda c: c["start"])
    return detected


def _hybrid_result_is_plausible(
    detected: List[Dict], anchors: List[Tuple[int, str]], used_markers: bool
) -> bool:
    """Sanity-check a complete hybrid result so a degraded one falls back to regex.

    The hybrid detector returns [] on total failure, but a parseable-but-wrong
    result (the LLM mislabels every anchor) would otherwise pass the caller's
    `len(chapters) >= 2` gate and drive chunking with bad chapter numbers. Two
    cheap checks catch the failure modes we've actually seen:

    - Story-order numbering: narrative chapter_numbers (non-reference, non-null),
      in start order, must equal 1..N exactly. This is the prompt's own contract
      ("chapter_number = its position in story order"), so a correct result always
      passes; it rejects duplicates, out-of-order labels, a wrong start, and the
      documented "LLM copied the number out of the anchor line" failure (numbering
      '=== Section 2/3 ===' as 2,3 instead of 1,2).
    - Marker coverage (marker-path only): when anchors are the extractor's
      '=== ... ===' section markers, dropping more than half of them signals gross
      over-omission. Skipped on the heading_candidates path, where anchors include
      heading-shaped noise the LLM legitimately omits.
    """
    narrative = [
        c["chapter_number"]
        for c in detected
        if not c["is_reference"] and c["chapter_number"] is not None
    ]
    if narrative != list(range(1, len(narrative) + 1)):
        return False

    if used_markers and len(detected) * 2 < len(anchors):
        return False

    return True


def detect_chapters_llm(
    content: str,
    invoke: Optional[Callable[[str], Dict]] = None,
) -> List[Dict]:
    """LLM-based chapter detector. Mirrors the regex detector's output shape.

    Returns a list of {"start", "title", "chapter_number", "is_reference"} dicts
    sorted by start offset. Returns [] on any failure so callers can fall back.

    `invoke` is an optional seam for tests: a callable taking the prompt and
    returning {"text": str, ...} (the shape of
    enhanced_rag_service.invoke_with_fallback). Production callers omit it.
    """
    try:
        candidates = heading_candidates(content)
        if not candidates:
            return []

        invoke_fn = invoke or _default_invoke
        response = invoke_fn(_build_prompt(candidates))
        text = response.get("text", "") if isinstance(response, dict) else str(response)

        parsed = _parse_llm_json_array(text)
        if not parsed:
            return []

        return _reconcile(parsed, candidates, content)

    except Exception:
        logger.exception("LLM chapter detection failed; returning empty result")
        return []


# ---------- Hybrid detector: deterministic anchors + one LLM labelling call ----------

def hybrid_anchors(content: str) -> List[Tuple[int, str]]:
    """Return (char_offset, line_text) anchors for the hybrid detector.

    Prefers the ingest extractor's `=== ... ===` section markers when at least
    two are present (epub uploads); otherwise falls back to heading_candidates()
    so PDF / plain-text documents still work.
    """
    markers = [(m.start(), m.group(0).strip()) for m in _MARKER_RE.finditer(content)]
    if len(markers) >= 2:
        return markers
    return heading_candidates(content)


def _snippet_after(content: str, offset: int) -> str:
    """First ~_SNIPPET_LEN chars of body text after the line at `offset`.

    Whitespace-collapsed so multi-line epigraphs read as one phrase. Empty
    string when the line is the last thing in the document.
    """
    line_end = content.find("\n", offset)
    if line_end == -1:
        return ""
    window = content[line_end : line_end + 5 * _SNIPPET_LEN]
    return " ".join(window.split())[:_SNIPPET_LEN]


def _build_hybrid_prompt(anchors: List[Tuple[int, str]], content: str) -> str:
    """Build the labelling prompt: each anchor line plus its following text."""
    numbered = "\n".join(
        f"{i}: {line}  >> {_snippet_after(content, off) or '(no following text)'}"
        for i, (off, line) in enumerate(anchors, start=1)
    )
    return (
        "You are labelling the section structure of an ingested book. Each line "
        "below is a section anchor, prefixed with a numeric ID; after '>>' come "
        "the first words of the text that follows that anchor in the book. "
        "Anchors like '=== Section 12 ===' are file-section markers stamped by "
        "our ebook extractor — they often mark real chapters, but their numbers "
        "count files, not chapters.\n\n"
        "Classify EVERY anchor as one of:\n"
        "- Front matter (table of contents, title/copyright page, dedication, "
        "lists of the author's other books): OMIT it from the output.\n"
        "- Narrative chapter: include it with chapter_number = its position in "
        "story order (1, 2, 3, ...). NEVER copy a number out of the anchor "
        "line itself.\n"
        "- Reference / back matter (appendix, glossary or terminology, maps or "
        "cartographic notes, afterword, about the author — and the sections "
        "that continue them): include it with chapter_number null and "
        "is_reference true.\n\n"
        "Return ONLY a JSON array, with no prose and no markdown fences:\n"
        '[{"id": <int>, "title": "<verbatim anchor line>", "chapter_number": '
        '<int|null>, "is_reference": <true|false>}, ...]\n\n'
        "Anchors:\n"
        f"{numbered}\n"
    )


def detect_chapters_hybrid(
    content: str,
    invoke: Optional[Callable[[str], Dict]] = None,
) -> List[Dict]:
    """Hybrid detector: regex-found anchors, LLM-labelled. One LLM call per doc.

    Boundary selection is NOT delegated to the LLM (measured boundary recall
    0.11 on a real book when it picks from bare heading lines): anchors come
    from hybrid_anchors(), and the LLM only classifies each anchor — front
    matter (omitted), narrative chapter (story-order numbering), or
    reference/back matter. Same output shape as the other detectors, sorted by
    start; [] on any failure so callers can fall back.
    """
    try:
        anchors = hybrid_anchors(content)
        if not anchors:
            return []

        invoke_fn = invoke or _default_invoke
        response = invoke_fn(_build_hybrid_prompt(anchors, content))
        text = response.get("text", "") if isinstance(response, dict) else str(response)

        parsed = _parse_llm_json_array(text)
        if not parsed:
            if _looks_truncated(text):
                logger.warning(
                    "Chapter labelling output looks truncated (%d chars, no "
                    "closing ']'); the LLM likely hit its output-token cap. Raise "
                    "LLM_CHAPTER_DETECTION_MAX_TOKENS and re-ingest. Falling back "
                    "to regex chapter detection.",
                    len(text),
                )
            return []

        detected = _reconcile(parsed, anchors, content)

        # A parseable-but-wrong result (mislabeled numbering / over-omission) would
        # otherwise pass the caller's `len >= 2` gate; treat it as a failure so the
        # caller falls back to regex, and say so loudly rather than silently.
        used_markers = len(_MARKER_RE.findall(content)) >= 2
        if not _hybrid_result_is_plausible(detected, anchors, used_markers):
            logger.warning(
                "Hybrid chapter labelling looks implausible (%d sections from %d "
                "anchors; narrative numbering not story-order 1..N or coverage too "
                "low). Falling back to regex chapter detection.",
                len(detected), len(anchors),
            )
            return []

        return detected

    except Exception:
        logger.exception("Hybrid chapter detection failed; returning empty result")
        return []
