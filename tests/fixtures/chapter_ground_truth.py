"""Labelled ground truth for the chapter-detection comparison.

Synthetic documents (no copyrighted text) with hand-labelled chapter structure.
Used by the comparison eval (tests/eval/run_chapter_eval.py) and the metric
tests to measure regex vs LLM chapter detection.

Each record:
    {
      "name": str,
      "text": str,                       # the document
      "expected": [                      # ground-truth chapters, sorted in order
        {"title": str,                   # verbatim heading line (offset derived by find())
         "chapter_number": int|None,
         "is_reference": bool},
        ...
      ],
    }

Buckets:
- "regex-handles": cases the existing regex already gets right (regression lock).
- "regex-fails": titled-only / unkeyworded headings the regex provably misses —
  this is the hypothesised win for the LLM approach.

Headings are separated by blank lines so the LLM detector's heading-candidate
filter (which keys on standalone short lines) can see them.
"""

GROUND_TRUTH = [
    # ---------- Bucket: regex-handles (regression lock) ----------
    {
        "name": "arabic_numbered",
        "text": (
            "Chapter 1\n"
            "The harvest began before dawn in the lowland fields.\n\n"
            "Chapter 2\n"
            "By midwinter the granaries stood nearly empty.\n"
        ),
        "expected": [
            {"title": "Chapter 1", "chapter_number": 1, "is_reference": False},
            {"title": "Chapter 2", "chapter_number": 2, "is_reference": False},
        ],
    },
    {
        "name": "numbered_with_glossary",
        "text": (
            "Chapter 1\n"
            "The council convened in the great hall at dusk.\n\n"
            "Chapter 2\n"
            "Word of the treaty reached the outer provinces.\n\n"
            "Glossary\n"
            "Terms and titles used throughout the chronicle.\n"
        ),
        "expected": [
            {"title": "Chapter 1", "chapter_number": 1, "is_reference": False},
            {"title": "Chapter 2", "chapter_number": 2, "is_reference": False},
            {"title": "Glossary", "chapter_number": None, "is_reference": True},
        ],
    },

    # ---------- Bucket: regex-fails (the hypothesised win) ----------
    {
        "name": "titled_only",
        "text": (
            "The Gathering Storm\n"
            "Clouds massed over the harbour as the fleet made ready.\n\n"
            "A Meeting at Dusk\n"
            "They spoke in low voices beneath the broken arch.\n\n"
            "The Long Road North\n"
            "Snow had closed the high passes weeks before.\n"
        ),
        "expected": [
            {"title": "The Gathering Storm", "chapter_number": 1, "is_reference": False},
            {"title": "A Meeting at Dusk", "chapter_number": 2, "is_reference": False},
            {"title": "The Long Road North", "chapter_number": 3, "is_reference": False},
        ],
    },
    {
        "name": "titled_with_afterword",
        "text": (
            "Embers\n"
            "The forge had gone cold for the first time in a generation.\n\n"
            "Ashfall\n"
            "Grey snow drifted across the abandoned square.\n\n"
            "Afterword\n"
            "A note on the sources behind this retelling.\n"
        ),
        "expected": [
            {"title": "Embers", "chapter_number": 1, "is_reference": False},
            {"title": "Ashfall", "chapter_number": 2, "is_reference": False},
            {"title": "Afterword", "chapter_number": None, "is_reference": True},
        ],
    },
]
