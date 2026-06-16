# Handoff — Multi-tenancy + Google auth (resume here)

**Branch:** `feat/multi-tenancy` · **Updated:** 2026-06-16 · **Status:** Increments 1 & 2 done,
106→109 tests green, security-reviewed. Increment 1 is committed (`92a3d1d`); **Increment 2 is
staged/uncommitted** at the time of writing (see the commit step).

Full design + decisions live in `current_state.md` → "Phase 2 — Increment 1 / Increment 2".
This file is the short "what to do when you sit back down".

---

## What's done

- **Increment 1 (committed):** `user_id` on docs, FAISS chunks, and conversation sessions;
  retrieval / list / delete / sessions all scoped to the owner, behind a stub `get_current_user`.
- **Increment 2:** real Google login — authlib + Starlette `SessionMiddleware` signed cookie.
  `/auth/login|callback|logout|me`; `get_current_user` reads the session (401 when absent), with
  a `DEV_AUTH_BYPASS` local-dev escape hatch. Frontend gates on `/auth/me`. `.env.example` added.
- **Security review (2026-06-16):** 7 confirmed findings, all fixed (Secure cookie flag via
  `SESSION_COOKIE_SECURE`; fail-closed on default secret; auth-gated `/chat/status` +
  `/documents/stats/overview`; `DEV_AUTH_BYPASS` startup warning; `oauth_sub` mismatch log;
  `IntegrityError` race handling; fail-closed `email_verified`).

---

## Resume steps

```bash
cd /home/dgbisme/code/D-G-B/GenAI-Book-Assistant

# 1. Tests
uv run pytest tests/                 # expect 109 passed

# 2. MANDATORY one-time DB reset (Increment 1 added a NOT NULL user_id column;
#    Increment 2 made User.email unique + added oauth columns — create_all can't
#    alter the old app.db, and old FAISS chunks predate user_id). Data is disposable.
rm -f app.db && rm -rf faiss_index/

# 3. Local run WITHOUT Google yet — set these in .env first:
#      DEV_AUTH_BYPASS=true       (logs you in as the dev user)
#    (Required: the app now refuses to start with no SESSION_SECRET_KEY unless
#     DEV_AUTH_BYPASS=true or DEBUG=true.)
uv run python -c "from main import app; print('OK')"
uv run uvicorn main:app --reload     # upload a book at http://localhost:8000

# 4. Commit Increment 2 (you push). Suggested message:
git add -A
git commit -m "feat: Google OAuth login (Increment 2) + session hardening from security review"
```

---

## To actually log in with Google (manual, free — only you can do this)

1. console.cloud.google.com → APIs & Services → **OAuth consent screen** → External; add your
   Google account under **Test users**.
2. **Credentials → Create OAuth client ID → Web application.** Authorized redirect URI:
   `http://localhost:8000/auth/callback`.
3. In `.env`: set `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, and a real
   `SESSION_SECRET_KEY` (`python -c "import secrets; print(secrets.token_hex(32))"`); unset
   `DEV_AUTH_BYPASS` (or set false).
4. Restart, click "Login with Google".

---

## What's left (next, in rough order — all from current_state.md)

- **Gate the still-global endpoints** if going multi-user for real: `POST /documents/{id}/process`,
  `POST /documents/rebuild-index`, `DELETE /documents/all` (the last wipes ALL users' docs).
- **Per-user stats**: `/chat/status` + `/documents/stats/overview` are auth-gated but counts are
  still instance-global.
- **Postgres + Alembic** (proper migration instead of the disposable SQLite reset), then
  **Dockerfile + deploy**, **CI**, **streaming responses** (Phase 2 items 3–6).
- Optional hardening: POST (not GET) logout + CSRF tokens on state-changing requests.
