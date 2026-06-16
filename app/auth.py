"""Authentication: Google OAuth + signed-cookie session.

`get_current_user` resolves the logged-in user from the session cookie
(`request.session["user_id"]`), 401 when absent. The OAuth client and the
find-or-create-by-email helper live here; the login/callback/logout/me routes
are in `app/api/auth_routes.py`.

Increment 1 introduced this module as a stub (fixed dev user). Increment 2
replaced the stub body with session resolution; the dev user survives only as
`get_or_create_dev_user`, used by the startup seed and the local-dev
`DEV_AUTH_BYPASS` escape hatch.
"""

import logging

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import User, get_db

logger = logging.getLogger(__name__)

DEV_USERNAME = "dev"
DEV_EMAIL = "dev@example.com"

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# OAuth registry. Google is registered only when credentials are present, so the
# app still imports and runs (e.g. with DEV_AUTH_BYPASS) without them.
oauth = OAuth()
if settings.GOOGLE_OAUTH_CLIENT_ID and settings.GOOGLE_OAUTH_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=settings.GOOGLE_OAUTH_CLIENT_ID,
        client_secret=settings.GOOGLE_OAUTH_CLIENT_SECRET,
        server_metadata_url=GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )


def google_oauth_configured() -> bool:
    """True when Google OAuth credentials are configured (login is possible)."""
    return bool(settings.GOOGLE_OAUTH_CLIENT_ID and settings.GOOGLE_OAUTH_CLIENT_SECRET)


def get_or_create_dev_user(db: Session) -> User:
    """Find or create the fixed development user (username='dev').

    Used by the startup seed (main.py lifespan) and the DEV_AUTH_BYPASS fallback
    so local dev / tests can run without Google credentials.
    """
    user = db.query(User).filter(User.username == DEV_USERNAME).first()
    if user is None:
        user = User(username=DEV_USERNAME, email=DEV_EMAIL)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Seeded dev user (id=%d)", user.id)
    return user


def find_or_create_user(
    db: Session, *, email: str, oauth_provider: str, oauth_sub: str | None
) -> User:
    """Find a user by verified email, or create one. Identity is the email.

    First login backfills oauth provenance onto the matched account; later logins
    log a warning if the provider's subject id changes (possible email reuse /
    account takeover at the provider). The insert is race-safe against the
    unique(email) constraint (two concurrent first-time logins).
    """
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, oauth_provider=oauth_provider, oauth_sub=oauth_sub)
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            # A concurrent first-time login won the unique(email) insert — use it.
            db.rollback()
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                raise
            return user
        db.refresh(user)
        logger.info("Created user id=%d via %s", user.id, oauth_provider)
        return user

    if oauth_sub and user.oauth_sub and user.oauth_sub != oauth_sub:
        # Same verified email, different provider subject — surface it loudly.
        logger.warning(
            "User %d (%s) logged in with a different oauth_sub (stored=%s, incoming=%s)",
            user.id, user.email, user.oauth_sub, oauth_sub,
        )
    elif oauth_sub and not user.oauth_sub:
        user.oauth_provider = oauth_provider
        user.oauth_sub = oauth_sub
        db.commit()
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Resolve the logged-in user from the session cookie.

    401 when there's no valid session — unless DEV_AUTH_BYPASS is set, in which
    case we fall back to the dev user (local dev / tests without Google creds).
    """
    user_id = request.session.get("user_id")
    if user_id is not None:
        user = db.query(User).filter(User.id == user_id).first()
        if user is not None:
            return user
        # Session points at a user that no longer exists — drop it.
        request.session.pop("user_id", None)

    if settings.DEV_AUTH_BYPASS:
        return get_or_create_dev_user(db)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )
