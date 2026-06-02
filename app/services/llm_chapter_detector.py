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
"""

import json
import logging
from typing import Callable, List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Heading heuristics: a "candidate" line is short, standalone, and not a sentence.
_MAX_HEADING_LEN = 80
_MAX_HEADING_WORDS = 12
_SENTENCE_ENDINGS = (".", ",", ";", ":")


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

        # Map returned ids back to authoritative candidate offsets. Using the id
        # avoids the classic bug where content.find(title) matches a copy of the
        # title inside body prose rather than the real heading.
        detected: List[Dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            offset: Optional[int] = None
            title = item.get("title")

            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                offset, title = candidates[idx - 1]
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

    except Exception:
        logger.exception("LLM chapter detection failed; returning empty result")
        return []
