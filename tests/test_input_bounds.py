"""Input-bound validation: the question-length cap on ChatRequest.

(The upload size limit and Query ge=1 bounds are enforced at the route layer and
exercised through the live app, not here.)
"""

import pytest
from pydantic import ValidationError

from app.config import settings
from app.schemas.chat import ChatRequest


def test_question_within_limit_ok():
    ChatRequest(question="a" * settings.MAX_QUESTION_LENGTH)


def test_question_over_limit_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(question="a" * (settings.MAX_QUESTION_LENGTH + 1))
