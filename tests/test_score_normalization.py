"""Tests for VectorStoreManager.normalize_score — L2-distance to cosine math.

For unit-normalized embeddings, ||a - b||^2 = 2 - 2 * cos(a, b),
so cos(a, b) = 1 - dist/2. Output is clamped to [0, 1].
"""

import pytest

from app.services.vector_store_manager import VectorStoreManager


@pytest.mark.parametrize(
    "raw_l2,expected_cosine",
    [
        (0.0, 1.0),   # identical vectors → cos=1
        (1.0, 0.5),   # 60° apart → cos=0.5
        (2.0, 0.0),   # orthogonal → cos=0
    ],
)
def test_normalize_score_within_range(raw_l2, expected_cosine):
    assert VectorStoreManager.normalize_score(raw_l2) == pytest.approx(expected_cosine)


def test_normalize_score_clamps_negative_drift_to_one():
    """Tiny floating-point drift below 0 (e.g. -1e-9) must clamp to 1.0."""
    assert VectorStoreManager.normalize_score(-0.0001) == 1.0


def test_normalize_score_clamps_above_orthogonal_to_zero():
    """L2 > 2 (opposite vectors) must clamp to 0.0."""
    assert VectorStoreManager.normalize_score(4.0) == 0.0
    assert VectorStoreManager.normalize_score(2.0001) == 0.0


def test_normalize_score_returns_float():
    assert isinstance(VectorStoreManager.normalize_score(1.0), float)
