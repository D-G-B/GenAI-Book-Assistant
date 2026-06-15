"""Build tests/eval/fixtures/dune_chapters.json — real-book chapter ground truth.

The fixture embeds the full extracted text of the Dune epub (copyrighted), so it
is gitignored and must be regenerated locally from Books/:

    uv run python -m tests.eval.build_dune_fixture

Labeling decisions (verified by inspecting every section of the 40th Anniversary
Ace epub as extracted by run_eval.extract_epub_text, which mirrors production
ingest). The extractor emits one `=== <title|Section N> ===` marker per epub
spine item; in this edition that yields 91 markers:

- 'Section 1'..'Section 4': front matter (ToC, ad page, copyright, dedication).
  Excluded from expected, matching the synthetic fixtures' convention.
- 'Section 5'..'Section 52': the 48 unnumbered narrative chapters (each opens
  with a Princess Irulan epigraph) -> chapter_number 1..48. Note: the Book
  One/Two/Three part headings do not survive extraction (image pages); they
  appear only in the ToC text.
- 'Appendix I' onward: back matter -> chapter_number null, is_reference true
  (Appendix I-III, 7 Appendix-IV noble-house entries, Terminology heading,
  25 glossary letter-groups, Cartographic Notes, Afterword, About the Author).

Explicit 'start' offsets are embedded because the ToC near offset 0 repeats
several headings, which would defeat resolve_offsets' first-match text.find().
"""
import json
import re
from pathlib import Path

from tests.eval.run_eval import extract_epub_text, find_default_book

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "dune_chapters.json"


def main():
    book = find_default_book()
    text = extract_epub_text(book)

    markers = [(m.start(), m.group(0))
               for m in re.finditer(r"^=== (.+?) ===$", text, re.MULTILINE)]
    # Guard against a different edition/extraction drifting under the labels.
    assert len(markers) == 91, f"expected 91 markers, got {len(markers)}"
    assert markers[4][1] == "=== Section 5 ===", markers[4]
    assert markers[51][1] == "=== Section 52 ===", markers[51]
    assert "Appendix I" in markers[52][1], markers[52]

    expected = []
    for n, (off, line) in enumerate(markers[4:52], start=1):
        expected.append({"title": line, "start": off,
                         "chapter_number": n, "is_reference": False})
    for off, line in markers[52:]:
        expected.append({"title": line, "start": off,
                         "chapter_number": None, "is_reference": True})

    fixture = {"name": f"dune-real-book ({book.name[:40]}...)",
               "text": text, "expected": expected}

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=1), encoding="utf-8")

    n_ch = sum(1 for e in expected if not e["is_reference"])
    print(f"chapters: {n_ch}, reference units: {len(expected) - n_ch}, "
          f"text chars: {len(text)}")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
