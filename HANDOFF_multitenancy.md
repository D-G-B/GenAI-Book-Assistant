# Handoff — Multi-tenancy Increment 1 (resume here)

**Branch:** `feat/multi-tenancy` · **Date paused:** 2026-06-16 · **Status:** Increment 1 code +
tests complete and green (98 passing). **Not committed yet.** Server needs a one-time DB reset
before it will run (see step 2).

The full design lives in `current_state.md` → "Phase 2 — Increment 1: Multi-tenancy
(IMPLEMENTED 2026-06-15)". This file is just the short "what to do when you sit back down".

---

## What's done (uncommitted, in the working tree)

Multi-tenancy is implemented and enforced behind a **stub** `get_current_user` (returns a fixed
dev user). Every document + FAISS chunk + conversation session is owned by a `user_id`;
retrieval / list / delete / session access are scoped to the owner.

Files changed:
- **`app/auth.py`** (new) — `get_or_create_dev_user` + `get_current_user` stub. *This is the
  only seam Increment 2 rewires.*
- `app/database.py` — `LoreDocument.user_id` FK (NOT NULL, indexed).
- `main.py` — seeds the dev user in lifespan.
- `app/services/document_manager.py` — stamps `user_id` into chunk `base_metadata`; `list`/`delete` scoped.
- `app/services/vector_store_manager.py` — `_build_filter_function` + `search_with_scores` reject foreign `user_id`.
- `app/services/enhanced_rag_service.py` + `app/services/conversational_memory.py` — thread `user_id` to retrieval; session ownership checks.
- `app/api/{chat,conversational,documents}_routes.py` — `Depends(get_current_user)`, pass `current_user.id`.
- Tests (new): `tests/test_user_filter.py`, `tests/test_user_scoping.py`, `tests/test_conversation_ownership.py`.
- Tests (touched): `tests/test_ingest_chapter_detection.py`, `tests/add_complex_document.py` (carry `user_id`).

---

## Resume steps

```bash
# 0. (WSL) cd into the repo
cd /home/dgbisme/code/D-G-B/GenAI-Book-Assistant

# 1. Confirm tests still green
uv run pytest tests/                # expect 98 passed

# 2. MANDATORY one-time DB reset (data is disposable; old app.db lacks the user_id
#    column and old FAISS chunks predate it). Then re-ingest a book via the UI.
rm -f app.db && rm -rf faiss_index/

# 3. Smoke-test boot + run the server
uv run python -c "from main import app; print('OK')"
uv run uvicorn main:app --reload    # upload a book at http://localhost:8000, confirm chat works

# 4. Commit (the user pushes). Suggested message:
git add -A
git commit -m "feat: multi-tenancy (Increment 1) — user_id on docs/chunks/sessions, enforced behind stub auth"
```

---

## Verification still owed (do after the DB reset)

- **Automated:** `uv run pytest tests/` (done above) + import boot (step 3).
- **Manual cross-user isolation:** the stub only has ONE dev user, so true two-user isolation
  can't be exercised until Increment 2. The cross-user logic IS covered by the new unit tests
  (filter rejects foreign `user_id`; list/delete/session scoping). If you want a live two-user
  check before auth, temporarily add an `X-User-Id` override to `get_current_user`.

---

## Next: Increment 2 — real Google login (authlib + cookie session)

Only `get_current_user` changes; everything above stays. Steps (from the plan):
1. Add `authlib` + `itsdangerous` to `pyproject.toml`.
2. Register Starlette `SessionMiddleware` in `main.py` (secret from env).
3. Register Google OAuth (authlib); add a checked-in `.env.example` (still missing) with Google + session vars.
4. New `app/api/auth_routes.py`: `/auth/login`, `/auth/callback` (find-or-create `User` by verified email → `request.session["user_id"]`), `/auth/logout`, `/auth/me`.
5. Rewire `get_current_user` to read `request.session["user_id"]` (401 when absent).
6. Frontend: "Login with Google" button + logout in `templates/index.html` / `static/js/app.js` (same-origin `fetch` sends the cookie automatically).
7. Google Cloud: OAuth consent + credentials; redirect URI `http://localhost:8000/auth/callback`.

**Blocker to decide when reached:** auth-flow integration tests need `TestClient`, currently
broken here (fastapi 0.104 + httpx 0.28 → `Client.__init__() got an unexpected keyword
argument`). Either pin `httpx<0.28` for the test group or bump fastapi/starlette.

**Also gate in Increment 2** (still global today): `POST /documents/{id}/process`,
`POST /documents/rebuild-index`, `DELETE /documents/all` (the last wipes *all* users' docs).
