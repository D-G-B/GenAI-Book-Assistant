"""Tests for Settings.validate_api_keys — key presence and key/model pairing.

A provider key without its matching DEFAULT_*_MODEL is silently skipped in
_initialize_llms, so validate_api_keys warns about it. Instance attributes are
overridden per-test (validate_api_keys reads self.*).
"""

import logging

from app.config import Settings


def make_settings(**overrides):
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_key_without_model_warns(caplog):
    s = make_settings(
        OPENAI_API_KEY="x", DEFAULT_OPENAI_MODEL=None,
        ANTHROPIC_API_KEY="x", DEFAULT_CLAUDE_MODEL="claude",
        GOOGLE_API_KEY="x", DEFAULT_GEMINI_MODEL="gemini",
    )
    with caplog.at_level(logging.WARNING):
        result = s.validate_api_keys()
    assert result is True  # all keys present
    assert "DEFAULT_OPENAI_MODEL is missing" in caplog.text


def test_missing_key_returns_false(caplog):
    s = make_settings(
        OPENAI_API_KEY=None, DEFAULT_OPENAI_MODEL="gpt",
        ANTHROPIC_API_KEY="x", DEFAULT_CLAUDE_MODEL="claude",
        GOOGLE_API_KEY="x", DEFAULT_GEMINI_MODEL="gemini",
    )
    with caplog.at_level(logging.WARNING):
        result = s.validate_api_keys()
    assert result is False
    assert "Missing API keys" in caplog.text


def test_complete_config_no_incomplete_warning(caplog):
    s = make_settings(
        OPENAI_API_KEY="x", DEFAULT_OPENAI_MODEL="gpt",
        ANTHROPIC_API_KEY="x", DEFAULT_CLAUDE_MODEL="claude",
        GOOGLE_API_KEY="x", DEFAULT_GEMINI_MODEL="gemini",
    )
    with caplog.at_level(logging.WARNING):
        assert s.validate_api_keys() is True
    assert "Incomplete provider config" not in caplog.text
