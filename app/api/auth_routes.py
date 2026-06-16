"""Google OAuth login routes (Increment 2).

Mounted at top-level `/auth` (not under `/api/v1`) so the callback URL matches the
redirect URI registered in Google Cloud: http://localhost:8000/auth/callback.

Flow: /auth/login → redirect to Google → /auth/callback (exchange code, verify
email, find-or-create user, set session) → redirect home. /auth/me reports the
current user (401 when logged out); /auth/logout clears the session.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth import (
    find_or_create_user,
    get_current_user,
    google_oauth_configured,
    oauth,
)
from app.database import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/login")
async def login(request: Request):
    """Start the Google OAuth flow."""
    if not google_oauth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured on this server.")
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    """Exchange the auth code, verify the email, log the user in."""
    if not google_oauth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured on this server.")

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:  # authlib raises on state mismatch / exchange failure
        logger.warning("OAuth callback failed: %s", exc)
        raise HTTPException(status_code=401, detail="OAuth authentication failed")

    userinfo = token.get("userinfo")
    if not userinfo or not userinfo.get("email"):
        raise HTTPException(status_code=401, detail="Google did not return an email address")
    # Fail closed: require an explicitly-truthy email_verified (a missing or
    # string-valued claim must NOT be treated as verified).
    if not userinfo.get("email_verified"):
        raise HTTPException(status_code=403, detail="Google email address is not verified")

    user = find_or_create_user(
        db,
        email=userinfo["email"],
        oauth_provider="google",
        oauth_sub=userinfo.get("sub"),
    )
    request.session["user_id"] = user.id
    logger.info("User %d (%s) logged in", user.id, user.email)
    return RedirectResponse(url="/", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    """Clear the session and return home."""
    request.session.pop("user_id", None)
    return RedirectResponse(url="/", status_code=303)


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    """Return the current user (401 when not logged in) — used to gate the frontend."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
    }
