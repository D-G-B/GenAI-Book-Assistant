"""Head-to-head: regex vs LLM chapter detection.

Scores both detectors against labelled ground truth (synthetic fixtures, plus
one hand-labelled real book if present) and writes JSON artifacts mirroring the
retrieval eval's baseline.json vs with_reranker.json proof.

Usage (from project root):
    # Deterministic, no API keys — exercises the harness + metric only:
    uv run python -m tests.eval.run_chapter_eval --mock-fixture

    # Real LLM run (needs an API key); add the Dune labels for the real-book row:
    uv run python -m tests.eval.run_chapter_eval

Artifacts:
    tests/eval/results/chapter_regex.json
    tests/eval/results/chapter_llm.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "tests" / "eval" / "results"
DUNE_LABELS = PROJECT_ROOT / "tests" / "eval" / "fixtures" / "dune_chapters.json"

from tests.fixtures.chapter_ground_truth import GROUND_TRUTH
from tests.eval.chapter_metrics import score_detection, aggregate


# ---------- Detector adapters ----------

def regex_detect(text: str) -> List[Dict[str, Any]]:
    from app.services.document_manager import DocumentManager
    dm = DocumentManager.__new__(DocumentManager)
    return dm._detect_chapters_in_content(text)


def llm_detect(text: str, invoke=None) -> List[Dict[str, Any]]:
    from app.services.llm_chapter_detector import detect_chapters_llm
    return detect_chapters_llm(text, invoke=invoke)


def make_mock_invoke(expected: List[Dict[str, Any]], text: str):
    """A canned invoke() that 'detects' exactly the ground truth for one doc.

    Lets --mock-fixture exercise the full harness + metric deterministically
    without keys. It maps each expected heading to its candidate id so the
    detector's id-reconciliation path is genuinely exercised.
    """
    from app.services.llm_chapter_detector import heading_candidates
    cands = heading_candidates(text)
    title_to_id = {t: i for i, (_, t) in enumerate(cands, start=1)}

    items = []
    for exp in expected:
        cid = title_to_id.get(exp["title"])
        if cid is None:
            continue
        items.append({
            "id": cid,
            "title": exp["title"],
            "chapter_number": exp["chapter_number"],
            "is_reference": exp["is_reference"],
        })
    payload = json.dumps(items)

    def _invoke(_prompt):
        return {"text": payload, "provider": "mock-fixture", "calls": 0}
    return _invoke


# ---------- Dataset ----------

def load_dataset() -> List[Dict[str, Any]]:
    docs = list(GROUND_TRUTH)
    if DUNE_LABELS.exists():
        data = json.loads(DUNE_LABELS.read_text(encoding="utf-8"))
        docs.append(data)  # {"name", "text", "expected": [...]}
        print(f"Loaded real-book labels: {data.get('name')}")
    else:
        print(f"(No real-book labels at {DUNE_LABELS}; synthetic fixtures only.)")
    return docs


# ---------- Run ----------

def run_detector(docs, detector_name: str, mock: bool) -> Dict[str, Any]:
    per_doc = []
    scores = []
    for doc in docs:
        text, expected = doc["text"], doc["expected"]
        if detector_name == "regex":
            predicted = regex_detect(text)
        else:
            invoke = make_mock_invoke(expected, text) if mock else None
            predicted = llm_detect(text, invoke=invoke)

        score = score_detection(predicted, expected, text)
        scores.append(score)
        per_doc.append({"name": doc["name"], "score": score})

    return {
        "timestamp": datetime.now().isoformat(),
        "detector": detector_name,
        "mode": "mock-fixture" if (mock and detector_name == "llm") else "live",
        "summary": aggregate(scores),
        "results": per_doc,
    }


def _llm_preflight() -> bool:
    """Return True if a live LLM call succeeds; print clear guidance if not."""
    from app.services.enhanced_rag_service import enhanced_rag_service
    try:
        enhanced_rag_service.invoke_with_fallback('Reply with the single word: ok')
        return True
    except Exception as exc:
        print(
            "\nABORTING: no LLM provider succeeded, so the LLM run would be all "
            "zeros and tell us nothing.\n"
            f"  Last error: {exc}\n\n"
            "  Check .env: a key must be valid AND the model id current. Common causes:\n"
            "   - Google: a 404 'model not found' means the key is fine but the model id\n"
            "     is stale. Try DEFAULT_GEMINI_MODEL=gemini-2.0-flash (no 'models/' prefix).\n"
            "   - OpenAI 401 'project archived' / Anthropic 401 'invalid x-api-key': dead keys.\n\n"
            "  (Existing chapter_llm.json was left untouched.)",
            file=sys.stderr,
        )
        return False


def main():
    parser = argparse.ArgumentParser(description="Regex vs LLM chapter detection")
    parser.add_argument("--mock-fixture", action="store_true",
                        help="Feed canned per-doc LLM output (deterministic, no API keys)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    docs = load_dataset()
    print(f"Scoring {len(docs)} documents\n")

    # Preflight: on a live run, confirm at least one LLM provider actually works
    # before scoring. Otherwise every doc silently returns [] and we'd write a
    # misleading llm=0.00 artifact that looks like "the LLM failed the task"
    # rather than "no LLM ever ran (auth/config problem)".
    if not args.mock_fixture and not _llm_preflight():
        return 1

    regex_out = run_detector(docs, "regex", mock=args.mock_fixture)
    llm_out = run_detector(docs, "llm", mock=args.mock_fixture)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "chapter_regex.json").write_text(
        json.dumps(regex_out, indent=2), encoding="utf-8")
    (args.results_dir / "chapter_llm.json").write_text(
        json.dumps(llm_out, indent=2), encoding="utf-8")

    rf1 = regex_out["summary"].get("boundary_f1", 0.0)
    lf1 = llm_out["summary"].get("boundary_f1", 0.0)
    rnum = regex_out["summary"].get("chapter_number_accuracy", 0.0)
    lnum = llm_out["summary"].get("chapter_number_accuracy", 0.0)

    print(f"boundary_f1:             regex {rf1:.2f}  ->  llm {lf1:.2f}")
    print(f"chapter_number_accuracy: regex {rnum:.2f}  ->  llm {lnum:.2f}")
    print(f"\nArtifacts written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
