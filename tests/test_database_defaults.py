"""Regression guard: created_at defaults must be callables, not frozen values.

`Column(..., default=datetime.now(timezone.utc))` evaluates the timestamp once at
import time, so every row would share the same created_at (≈ server start). The fix
passes a callable (`lambda: datetime.now(timezone.utc)`) so SQLAlchemy calls it per
insert. These assertions fail on the old scalar default and need no database.
"""

import pytest

from app.database import LoreDocument, LoreQuery, User


@pytest.mark.parametrize("model", [User, LoreDocument, LoreQuery])
def test_created_at_default_is_callable(model):
    default = model.__table__.c.created_at.default
    assert default is not None
    assert default.is_callable, (
        f"{model.__name__}.created_at default must be a callable so it is evaluated "
        "per row, not frozen at import time"
    )
