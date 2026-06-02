"""Accuracy metrics for chapter-structure detection.

Scores a predicted chapter list against ground truth. The three fields measured
are exactly the ones the spoiler filter rides on: where a chapter starts
(boundary), its chapter_number, and whether it is reference material.

Predicted/expected items are dicts of the shape produced by the detectors:
    {"start": int, "title": str, "chapter_number": int|None, "is_reference": bool}

Ground-truth fixtures may omit "start" (offsets are derived by locating the
title in the source text); see `resolve_offsets`.
"""

from typing import Any, Dict, List

# Boundary match tolerance in characters. Chunkers only need approximate
# boundaries, so exact-equal would be needlessly brittle.
BOUNDARY_TOLERANCE = 5


def resolve_offsets(expected: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Fill in 'start' for expected chapters by locating each title in `text`.

    Keeps fixtures readable (they list titles, not offsets). Raises if a labelled
    title can't be found — that's a fixture authoring error, not a model result.
    """
    resolved = []
    for item in expected:
        if "start" in item:
            resolved.append(dict(item))
            continue
        offset = text.find(item["title"])
        if offset == -1:
            raise ValueError(f"Ground-truth title not found in text: {item['title']!r}")
        resolved.append({**item, "start": offset})
    return resolved


def _match_pairs(predicted, expected):
    """Greedily pair predicted↔expected chapters by boundary proximity.

    Each expected chapter is matched to at most one predicted chapter whose start
    is within BOUNDARY_TOLERANCE. Returns the list of (pred, exp) matched pairs.
    """
    remaining = list(predicted)
    pairs = []
    for exp in expected:
        best = None
        best_dist = BOUNDARY_TOLERANCE + 1
        for pred in remaining:
            dist = abs(pred["start"] - exp["start"])
            if dist <= BOUNDARY_TOLERANCE and dist < best_dist:
                best, best_dist = pred, dist
        if best is not None:
            pairs.append((best, exp))
            remaining.remove(best)
    return pairs


def score_detection(
    predicted: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
    text: str,
) -> Dict[str, float]:
    """Score predicted chapters against expected. See module docstring.

    Returns boundary precision/recall/f1 plus chapter_number and is_reference
    accuracy (both computed over correctly-bounded matches only — you can't grade
    a field on a chapter you never located).
    """
    expected = resolve_offsets(expected, text)
    pairs = _match_pairs(predicted, expected)

    n_pred = len(predicted)
    n_exp = len(expected)
    n_match = len(pairs)

    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_exp if n_exp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    num_correct = sum(1 for p, e in pairs if p.get("chapter_number") == e.get("chapter_number"))
    ref_correct = sum(1 for p, e in pairs if bool(p.get("is_reference")) == bool(e.get("is_reference")))

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "chapter_number_accuracy": (num_correct / n_match) if n_match else 0.0,
        "is_reference_accuracy": (ref_correct / n_match) if n_match else 0.0,
        "matched": n_match,
        "predicted": n_pred,
        "expected": n_exp,
    }


def aggregate(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro-average a list of per-document score dicts."""
    if not scores:
        return {}
    keys = ["boundary_precision", "boundary_recall", "boundary_f1",
            "chapter_number_accuracy", "is_reference_accuracy"]
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}
