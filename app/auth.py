"""Authentication seam for multi-tenancy.

Increment 1 (multi-tenancy): `get_current_user` returns a fixed dev user so every
ingest/retrieval path can be scoped by `user_id` before real login exists. This
dependency is the ONLY place Increment 2 (Google OAuth + cookie session) rewires:
it will resolve the user from `request.session["user_id"]` (401 when absent)
instead of returning the dev user. Nothing downstream changes — callers only
depend on this returning a `User`.
"""

import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import User, get_db

logger = logging.getLogger(__name__)

DEV_USERNAME = "dev"
DEV_EMAIL = "dev@example.com"


def get_or_create_dev_user(db: Session) -> User:
    """Find or create the fixed development user (username='dev').

    Shared by the startup seed (main.py lifespan) and `get_current_user` so the
    dev user row exists exactly once and its id is stable across requests.
    """
    user = db.query(User).filter(User.username == DEV_USERNAME).first()
    if user is None:
        user = User(username=DEV_USERNAME, email=DEV_EMAIL)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Seeded dev user (id=%d)", user.id)
    return user


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Resolve the current user. Increment 1 stub: always the dev user.

    Increment 2 replaces this body with session-cookie resolution; the signature
    (returns a `User`) is the stable contract the routes depend on.
    """
    return get_or_create_dev_user(db)
