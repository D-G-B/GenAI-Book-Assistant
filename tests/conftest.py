"""Pytest configuration for the test suite.

The legacy `test_conversational.py` and `test_reading_partner.py` are manual
scripts that POST to a live server on localhost:8000. Skip them under pytest;
they should be invoked directly with `uv run python tests/test_*.py`.
"""

collect_ignore_glob = [
    "test_conversational.py",
    "test_reading_partner.py",
    "add_complex_document.py",
]
