# Current State

A snapshot of where the project sits at the end of the Phase 1 hardening pass.

## Branch

**Active: `feat/multi-tenancy`** (forked from `main` for Phase 2 — see "Recommended next move"
below). The Phase 1 hardening from `claude/upbeat-lovelace-a4qv7` was merged to `main` via
**PR #3** (`bd81d14`), so `main` now contains everything below.

Phase 1 work (2026-06-15), on the now-merged `claude/upbeat-lovelace-a4qv7` — hybrid chapter
detector wired into ingest + two sanity sweeps (hardening, dead-code removal, the
parseable-but-wrong fallback, the per-row `created_at` fix):

```
81742d4 fix: model-name validation, dead-code cleanup, startup failure logging
0551538 doc: mark should_rebuild / upload-limit / chat-bounds fixes in current_state.md
c51a1e6 fix: should_rebuild ratio + upload size limit + chat input bounds
699d9cf doc: mark hybrid chapter detector wired into ingest; record provider findings
2bba9e6 feat: warn (not silently fall back) when chapter labelling is truncated
7b78003 feat: use hybrid LLM chapter detector at ingest, regex as fallback
```
(plus the doc commit recording these sanity-sweep fixes)

Prior session (2026-06-12):

```
ffdf97b feat: hybrid chapter detector - regex anchors + one small LLM labelling call
d659dd7 eval: hand-labeled Dune real-book fixture + live regex-vs-LLM results
6d6bb3e feat: support OPENAI_BASE_URL for GitHub Models PAT
48f4d6b eval: abort chapter eval loudly when no LLM provider works
32fc244 experiment: standalone LLM chapter detector + head-to-head eval vs regex
```

## What was hardened

### Correctness (Phase 1A)
- Real cosine-style similarity scores returned per source — no more hardcoded `0.85`.
- `k` argument honored in retrieval (was being silently widened to 20).
- `ChatAnthropic` import fixed to `langchain_anthropic` (was using deprecated `langchain_community.chat_models`).
- Bogus `claude>=0.4.11` package removed from `pyproject.toml` (it's not the Anthropic SDK).
- `uvicorn --reload` gated on `settings.DEBUG`.
- LLM clients now have explicit timeout + retry config.

### Observability (Phase 1B.1)
- Project-wide structured logging via `logging.basicConfig` in `main.py`, level from `settings.LOG_LEVEL`.
- All `print()` calls in `app/` replaced with module-level loggers.
- `traceback.print_exc()` calls replaced with `logger.exception(...)`.
- Bare `except:` clauses tightened to specific exception types.

### Tests (Phase 1B.2)
- `pytest` + `pytest-asyncio` + `httpx` added as dev deps; `[tool.pytest.ini_options]` configured.
- 41 unit tests, runs in ~1 second, no API keys required.
- Coverage focused on the IP: spoiler filter (22 tests), chapter extraction (13), score normalization (6).
- Legacy live-server scripts (`test_reading_partner.py`, `test_conversational.py`) excluded from pytest collection but still runnable directly.

### Retrieval eval (Phase 1B.3)
- 10 hand-authored Dune Q/A pairs at `tests/eval/dune_qa.jsonl`.
- `tests/eval/run_eval.py` builds a FAISS index from a book in `Books/`, runs queries, computes recall@5 over keyword containment.
- FAISS index cached at `tests/eval/.cache/` (gitignored) so re-runs are fast.
- Baseline saved to `tests/eval/results/baseline.json`.

### Retrieval quality (Phase 1C)
- **LCEL refactor of conversational chain** (1C.1): replaced deprecated `ConversationalRetrievalChain` with a direct condense → retrieve-with-scores → answer flow. Conversational path now returns real `similarity_score` and `confidence` (were `None`).
- **Cross-encoder reranker** (1C.2): `BAAI/bge-reranker-base`, lazy-loaded. Pulls `RERANK_POOL_SIZE=30` candidates from FAISS, reorders via cross-encoder, returns top-k. Toggle with `RERANKER_ENABLED`.
- **Batch embedding writes** (1C.3): `add_documents` now makes a single batched embedding call per upload instead of one call per chunk. ~10–50× faster on uploads after the first one. Per-chunk fallback if the batched call raises.

## Verified metrics

| Check | Result |
|---|---|
| `uv run pytest tests/` | 41/41 in ~1s, no API keys |
| Server boot | `from main import app` clean |
| Recall@5 baseline (no reranker) | 0.80 (8/10) |
| Recall@5 with reranker | 0.90 (9/10) |
| `grep -r "print(" app/` | empty |

## Known issues (bugs to fix)

- ✅ **RESOLVED (2026-06-15) — Titled-only chapters silently broke spoiler protection.**
  (Implementation + new findings under "Status" below.) `_detect_chapters_in_content`
  (`document_manager.py:231`) relies on regex patterns that require the word "Chapter" (or
  equivalent) in the heading. Books that use title-only chapter headings (e.g. "The Gathering
  Storm", "A Meeting at Dusk") return `< 2` matches, causing `_process_and_chunk` to fall
  through to `_chunk_flat` (line 418), which stamps `chapter_number=1` on every chunk. The
  spoiler filter then treats the whole book as chapter 1 — silently allowing any chapter's
  content to appear regardless of the user's `max_chapter` setting.
  **Planned fix:** evaluate a PageIndex-style LLM-based heading extractor as a smarter
  alternative (see plan in `.claude/plans/`); if that doesn't resolve it, patch the regex to
  accept standalone title-cased lines as chapter boundaries.
  **Status (2026-06-12): measured live, hybrid detector wins outright.** Three detectors
  scored against synthetic fixtures plus a hand-labelled real book (Dune 40th-Anniversary
  epub: 48 narrative chapters + 39 reference units; fixture gitignored — regenerate via
  `tests/eval/build_dune_fixture.py`). Real-book row:

  | Dune row | regex | LLM-solo | hybrid |
  |---|---|---|---|
  | boundary_f1 | 0.90 | 0.18 | **1.00** |
  | chapter_number_accuracy | 0.11 | 1.00 (on matches) | **1.00** |
  | is_reference_accuracy | 0.55 | 1.00 (on matches) | **1.00** |

  Regex finds boundaries only via the epub extractor's `=== Section N ===` artifacts and
  mis-numbers everything (+4 shift, zero reference flags); LLM-solo labels perfectly but
  can't tell which bare heading lines are chapters (it picks the ToC). The **hybrid**
  (`detect_chapters_hybrid`: regex-found marker anchors + ONE small LLM call that labels
  each anchor seeing a ~120-char snippet of following text) scored 1.00 on every metric,
  live, via GitHub Models gpt-4o-mini — ~5k tokens per book (~$0.002/book), 10x smaller
  than an LLM-solo prompt and the only variant that fits the PAT's 8k request cap.

  **Status (2026-06-15): WIRED INTO INGEST** (commits 7b78003 + 2bba9e6). In
  `DocumentManager._process_and_chunk`, `detect_chapters_hybrid(content)` runs first (off
  the event loop via `run_in_executor`) and chunks with its labels when it returns >= 2
  chapters; otherwise it falls through to the regex detector (behavior preserved — hybrid
  returns `[]` on LLM failure). Real Dune ingest now yields story-order chapters **1..48**
  (was 2..91) with reference material flagged. Suite 59 → 65, no API keys.

  Prereqs handled: added `LLM_CHAPTER_DETECTION_ENABLED` (default True) opt-out flag and
  `LLM_CHAPTER_DETECTION_MAX_TOKENS` (default **16000**, see below) in `config.py`;
  `invoke_with_fallback` gained a per-call `max_tokens` override applied via `.bind()`
  (NOT `.copy()`, which drops default-valued fields like `callbacks` on these
  pydantic-v1-shim models) using helper `_output_cap_kwargs` — `max_tokens` for
  OpenAI/Anthropic, `generation_config={"max_output_tokens": N}` for Google.

  Design kept as decided 2026-06-12: **no "zero-cost tier"** (skipping the LLM when regex
  finds `Chapter N` was rejected — sequel-preview "Chapter 1" sections and part-relative
  numbering leak through `max_chapter` filtering; only the LLM pass classifies front/back
  matter correctly).

  **New live findings (2026-06-15) that changed the eval-time picture:**
  - **gemini-2.5-flash (first in fallback) is a thinking model** — its reasoning consumes
    the output-token budget, so the labelling JSON truncates below ~16k. Hence the 16000
    default (scored 1.00 on Dune at 16000; truncates → regex fallback below it). Dune used
    ~15.5k, so headroom is thin for much larger books — a WARNING now fires on truncation
    naming the env var to raise.
  - **The GitHub Models PAT (gpt-4o-mini) drifted**: scored 1.00 on 2026-06-12 but 0.17 on
    2026-06-15 (confirmed stable across 3 runs — over-omits sections). Don't trust a past
    PAT score to hold; re-measure. Fine for short Q&A.
  - `gemini-1.5-flash` (404) and `gemini-2.0-flash` (empty) are unavailable on the current
    free key, so cheap non-thinking Gemini isn't an option here yet.

  Remaining follow-ups:
  - ✅ **DONE (2026-06-15, later) — *parseable-but-wrong* fallback gap closed.**
    `detect_chapters_hybrid` now sanity-checks a complete result before returning it
    (`_hybrid_result_is_plausible`): narrative `chapter_number`s must be story-order
    `1..N` exactly (rejects duplicates, out-of-order labels, and the documented failure
    of the LLM copying a number out of the `=== Section N ===` anchor line), and on the
    marker path dropping more than half the `=== ... ===` markers is rejected as gross
    over-omission. On failure it logs a WARNING and returns `[]`, so the existing regex
    fallback fires unchanged (`_process_and_chunk`'s `len(chapters) < 2` gate is
    untouched). **Residual gap:** a perfectly *renumbered* tiny subset on the
    `heading_candidates` path (PDF/plain text) still passes, since coverage is only
    checked on the marker path where anchors are reliable. Tests in
    `test_chapter_extraction_llm.py` + `test_ingest_chapter_detection.py`.
  - Recover the PAT as a reliable second free endpoint via a sharper detection prompt
    ("omit ONLY title page + ToC; never drop a narrative chapter").
  - Production cost (the only waste is ~13k thinking tokens/book): upgrade
    `langchain-google-genai` (pinned 1.0.4, predates Gemini 2.5) to control
    `thinking_budget`, or point detection at a verified cheap non-thinking model.
  - Optional regex hardening (independent): add Foreword/Preface/Prologue/Introduction to
    `reference_patterns`, and note the pattern-priority quirk (a titled `=== ... ===` match
    within 20 chars beats reference classification — why Dune's appendices weren't flagged).

The following were found in a codebase sanity sweep (2026-06). Most were fixed on 2026-06-15
(commits c51a1e6 + a follow-up pass, marked ✅ below); the nullable-columns item is already
guarded in code and its schema constraint is deferred (see its note):

- ✅ **FIXED (2026-06-15) — CRITICAL `should_rebuild()` ratio formula was wrong**
  (`vector_store_manager.py`). The old `deleted_count / (deleted_count + 10)` ignored index
  size and mis-fired; soft-deleted chunks accumulated in FAISS indefinitely. Now compares
  deleted *chunks* / total chunks — counted from the docstore, since `deleted_document_ids`
  tracks documents (one document maps to many chunks), so the doc's earlier one-line
  suggestion `deleted_count / total_chunks` would have mixed units. Tests in
  `tests/test_should_rebuild.py`.
- ✅ **FIXED (2026-06-15) — `validate_api_keys()` didn't validate model names** (`config.py`).
  Now warns at startup when a provider key is set without its matching `DEFAULT_*_MODEL`
  (such a provider is silently skipped in `_initialize_llms`). Tests in
  `tests/test_config_validation.py`.
- ✅ **FIXED (2026-06-15) — No file size limit on upload** (`documents_routes.py`). Reads at
  most `MAX_UPLOAD_SIZE_MB`+1 bytes and returns 413 if exceeded, so an oversized file is
  rejected without being fully loaded into memory.
- ✅ **FIXED (2026-06-15) — Unsupported file types "junk content"** (`documents_routes.py`).
  The unsupported-extension 400 (the `supported` set check) already runs before the read, so
  the `[Unsupported file type: ...]` branch was unreachable dead code — removed it.
- ✅ **FIXED then SUPERSEDED (2026-06-15) — `JSONLoader(text_content=False)`**
  (`advanced_document_loaders.py`) → `text_content=True`, then **the whole module was
  removed.** `DocumentProcessor` / `MultiFormatDocumentLoader` / `EpubLoader` /
  `WebDocumentLoader` were a second, diverging copy of the file-format extractors and
  entirely unused by the active ingest path — the upload route does its own inline
  PDF/Word/EPUB extraction in `documents_routes.py` (`_extract_pdf_content`,
  `_extract_word_content`, `_extract_epub_content`); `.json`/`.csv`/text are decoded as raw
  text. The only link was a dangling import + an assigned-but-never-read
  `self.document_processor` in `document_manager.py`, both removed. `advanced_document_loaders.py`
  is deleted, so the latent JSONLoader fix went with it (it was never on the active path).
- ✅ **FIXED (2026-06-15) — No input bounds on chat** (`chat_routes.py`, `schemas/chat.py`).
  Question length capped at `MAX_QUESTION_LENGTH` (422 on overflow); `max_chapter` and
  `document_id` query params require `ge=1`. Test in `tests/test_input_bounds.py`.
- ⏸️ **Nullable DB columns** (`database.py`). Re-examined 2026-06-15: the ingest path does
  NOT deref unchecked — `add_document` guards `content` and `_process_and_chunk` defaults
  `source_type` to `'text'`. Adding `nullable=False` now is a no-op on the existing SQLite DB
  (`create_all` won't alter it) and would desync from `DocumentCreate(content: Optional)`.
  Deferred to the Phase 2 Postgres/Alembic migration, where the constraint can be applied and
  the API schema aligned. (`doc_metadata` is defined-but-unused — harmless future field.)
- ✅ **FIXED (2026-06-15) — Startup reprocessing swallowed failure reasons** (`main.py`). The
  boot loop now checks `add_document`'s return value and logs a per-document warning on failure.
- ✅ **FIXED (2026-06-15) — Unused `BaseLoader` import** (`advanced_document_loaders.py`) —
  removed.

### Second sanity sweep (2026-06-15, pre-auth)

A read-through before starting auth/multi-tenancy surfaced one live bug, one bounds gap,
and a layer of dead code left after `DocumentProcessor` was removed. Fixed the live items
and removed the clearly-dead Python; the dead DB *tables* are deferred (see below).

- ✅ **FIXED — `created_at` default evaluated once at import** (`database.py`).
  `Column(DateTime(timezone=True), default=datetime.now(timezone.utc))` passed a value
  computed at import time, so every row shared one timestamp (≈ server start) — user-visible
  via `DocumentResponse.created_at`. Now `default=lambda: datetime.now(timezone.utc)` on
  `User` / `LoreDocument` / `LoreQuery`. Regression guard: `tests/test_database_defaults.py`
  (asserts the column default `is_callable`).
- ✅ **FIXED — Conversational route missing `ge=1` bounds** (`conversational_routes.py`).
  The earlier chat-input-bounds fix covered `chat_routes.py` but not the conversational
  endpoint; `max_chapter` / `document_id` now require `ge=1` there too. Verified live (the
  route-level bound isn't unit-tested — a `TestClient` test can't run in this env: the
  installed httpx/starlette versions are incompatible, `Client.__init__() got an unexpected
  keyword argument`).
- ✅ **REMOVED — dead Python** (no callers, confirmed by grep): `ChatHistory` schema
  (`schemas/chat.py`, also the source of the pydantic `model_used` protected-namespace
  warning), `max_chunks` field on `ChatRequest`, `DocumentChunk` / `DocumentChunkBase`
  schemas (`schemas/documents.py`), `VectorStoreManager.get_retriever()` (superseded by
  `search_with_scores`) and `.undelete_document()` (restore is done inline via
  `deleted_document_ids.discard()` in `document_manager.add_document`).
- ⏸️ **Deferred to the Postgres/Alembic migration:** the dead DB *tables* `DocumentChunk`
  (+ `LoreDocument.chunks` relationship) and `LoreQuery`, and the unused `doc_metadata`
  column. The migration rewrites the schema anyway, so that's where to decide whether they
  become real (persist chunks / log queries) or get dropped. Chunks currently live only in
  FAISS; no query logging exists.

Suite: 81 → **84**, ~1.7s, no API keys.

## Known quirks (deliberately left)

These are documented behaviors, not bugs:

- **Chapter detector dedupes markers within 20 chars**. If two chapter markers are very close together (rare in real books, common in tiny test fixtures), only the first is kept. Documented in `tests/test_chapter_extraction.py`.
- **Reranker preserves the FAISS L2 score** in the returned tuple. The reorder is the win; the displayed cosine similarity may not be monotonically decreasing after rerank. This is intentional — it keeps `normalize_score()` semantics stable for API consumers.
- **One eval question still misses** ("Who is the Padishah Emperor"). The Emperor is referred to as "the Emperor" far more often than "Shaddam"/"Corrino" in the book. This is a keyword-side issue, not a retrieval problem.
- **Pre-existing pydantic warning** about `model_used` field in `ChatHistory` schema conflicting with `model_` namespace. One-line fix (`model_config['protected_namespaces'] = ()`) but out of scope for Phase 1.
- **SQLAlchemy 1.x `declarative_base()` deprecation warning** — out of scope.

## What this codebase still doesn't do

Honest list of things absent today, ordered by how much they matter for going public:

### Multi-tenancy / auth
- ✅ **Increment 1 done (2026-06-15):** `user_id` now lives on `LoreDocument`, on every
  FAISS chunk's metadata, and on conversation sessions; retrieval / list / delete / session
  access are all scoped to the owner. Enforced behind a **stub** `get_current_user` (fixed dev
  user) — see "Phase 2 — Increment 1" below.
- ✅ **Increment 2 done (2026-06-16):** real Google login (authlib + Starlette
  `SessionMiddleware` signed cookie). `get_current_user` now resolves
  `request.session["user_id"]` (401 when absent); a `DEV_AUTH_BYPASS` env flag falls back to
  the dev user for local dev. See "Phase 2 — Increment 2" below.
- Admin/maintenance endpoints are still global (not user-scoped): `POST /documents/{id}/process`,
  `POST /documents/rebuild-index`, `DELETE /documents/all`. `delete_all` wipes **all** users'
  documents — a real cross-tenant footgun to gate in Increment 2.

### Deploy
- No Dockerfile.
- No CI (no GitHub Actions, no automated test run on PR).
- SQLite-only (`./rag_assistant.db`); no migrations (no Alembic).
- FAISS is in-process and on-disk; doesn't scale past one container.

### Robustness
- No rate limiting.
- Upload size (`MAX_UPLOAD_SIZE_MB`) and question length (`MAX_QUESTION_LENGTH`) are now
  bounded; no other request size limits beyond FastAPI defaults.
- No `.env.example` checked in.
- No structured request IDs / correlation IDs in logs.

### UX / product
- Sessions are in-memory (lost on restart).
- No streaming responses (LLM output blocks until complete).
- No way for users to preview chunks before sending a query.
- Frontend is functional but unstyled beyond the basics.

### Retrieval
- Character-based chunking (1000/200). Token-aware chunking would respect sentence boundaries better.
- MiniLM embeddings (384-dim). Newer models like `nomic-embed-text-v1.5` or `bge-m3` are stronger.
- No BM25 hybrid retrieval (lexical + semantic).
- No query rewriting beyond pronoun resolution.
- No answer faithfulness checks (does the LLM answer match the cited chunks?).

## Recommended next move (Phase 2 sketch)

**Status (2026-06-16): Increments 1 (multi-tenancy) AND 2 (real Google login) are both
IMPLEMENTED on `feat/multi-tenancy`.** Decisions locked and delivered: multi-tenancy first
behind a stub `get_current_user`, then real Google login via **authlib + a signed-cookie
session** — NOT `fastapi-users` (see the corrected recommendation below). The full
step-by-step handoff is the multi-tenancy + auth plan in `.claude/plans/`.

### Phase 2 — Increment 1: Multi-tenancy (IMPLEMENTED 2026-06-15)

Every document and retrieved chunk is now owned by a user; retrieval never returns another
user's chunks. Auth is stubbed — `get_current_user` returns a fixed dev user — so the only
seam Increment 2 rewires is that one dependency.

- **Auth seam** (`app/auth.py`, new): `get_or_create_dev_user(db)` (find-or-create
  `username="dev"`) + `get_current_user` dependency returning it. Increment 2 replaces only
  `get_current_user`'s body (resolve `request.session["user_id"]`, 401 when absent).
- **Model** (`app/database.py`): `LoreDocument.user_id = Column(Integer,
  ForeignKey("users.id"), nullable=False, index=True)`. Dev user seeded in `main.py` lifespan.
- **Stamp at ingest**: both upload routes set `LoreDocument.user_id = current_user.id`;
  `_process_and_chunk`'s `base_metadata` carries `user_id` into every FAISS chunk.
- **Enforce at retrieval (core)**: `VectorStoreManager._build_filter_function` gained a
  `user_id` param and rejects any chunk whose `metadata["user_id"]` differs (right after the
  soft-delete check). Threaded through `search_with_scores` ← `ask_question` /
  `ask_with_context` ← the chat / conversational routes (`user_id=current_user.id`).
- **Scope management**: `list_all_documents` / `delete_document` filter `LoreDocument` by
  `user_id`; the `/documents/{id}/status` route filters its query by owner.
- **Conversation ownership**: sessions bind to `current_user.id`; `get_history` /
  `clear_session` / `list_sessions` only touch the caller's sessions (`_owned_session`;
  non-owned == absent). `clear_conversation` now returns `False` for missing/foreign sessions,
  making the route's previously-dead 404 branch live.
- **Tests**: `tests/test_user_filter.py` (filter rejects foreign / unstamped chunks),
  `tests/test_user_scoping.py` (ingest stamps `user_id`; list/delete user-scoped, in-memory
  SQLite), `tests/test_conversation_ownership.py`. Suite **84 → 98**, ~0.6s, no API keys.
- **⚠️ MANDATORY DB RESET before running the server.** `create_all` can't add the new
  NOT-NULL `user_id` column to the existing `app.db` (old schema), and the on-disk FAISS chunks
  predate `user_id` (so they'd be blocked by the filter anyway). Data is disposable at this
  stage — delete and re-ingest:
  ```bash
  rm -f app.db && rm -rf faiss_index/
  ```
  Proper migration is deferred to the Postgres/Alembic move (Phase 2 item 3).
- **Known gaps left for Increment 2** (deliberate, documented above): `process_document` /
  `rebuild_index` / `delete_all_documents` routes are still global; `_fetch_k_for` isn't
  user-aware (moot under the single dev user); the removed free-form `user_id` query param on
  `/conversation/ask` is now derived from the authenticated user (FastAPI ignores a stray
  `user_id=` query arg, so the same-origin frontend is unaffected).

### Phase 2 — Increment 2: Real Google login (IMPLEMENTED 2026-06-16)

authlib Google OAuth + a Starlette `SessionMiddleware` signed cookie. Only the
`get_current_user` seam from Increment 1 changed behaviour; every tenancy rule above is
untouched.

- **Deps** (`pyproject.toml`): added `authlib>=1.3`, `itsdangerous>=2.1`; **pinned
  `httpx>=0.27,<0.28`** (main + dev). This both satisfies authlib's runtime httpx and
  **unblocks `TestClient`** — starlette 0.27's TestClient passes `app=` to `httpx.Client`,
  which 0.28 removed. (Installed: authlib 1.7.2, itsdangerous 2.2.0, httpx 0.27.2.) Bumping
  starlette/fastapi instead would ripple through the langchain pins, so httpx was the surgical fix.
- **Session** (`main.py`): `SessionMiddleware(secret_key=SESSION_SECRET_KEY, same_site="lax",
  https_only=settings.SESSION_COOKIE_SECURE)`. Startup **fails closed** when the secret is
  unset/default unless `DEBUG` or `DEV_AUTH_BYPASS` is set; `DEV_AUTH_BYPASS` logs a loud warning.
- **Config** (`config.py`): `SESSION_SECRET_KEY`, `SESSION_COOKIE_SECURE` (Secure cookie flag,
  default False — set true on HTTPS), `GOOGLE_OAUTH_CLIENT_ID`/`_SECRET`, `DEV_AUTH_BYPASS` (default False).
- **OAuth + seam** (`app/auth.py`): `oauth.register("google", server_metadata_url=<OIDC
  discovery>, scope="openid email profile")` only when creds are present; `find_or_create_user`
  (identity = verified email; backfills oauth provenance); `get_current_user(request, db)` reads
  `request.session["user_id"]` → 401, or the dev user when `DEV_AUTH_BYPASS`.
- **Routes** (`app/api/auth_routes.py`, mounted at **top-level `/auth`** so the callback matches
  Google's redirect URI): `/auth/login`, `/auth/callback` (verifies `email_verified`, sets
  session, 303 home), `/auth/logout`, `/auth/me` (401 when logged out — the frontend's gate).
- **Model** (`database.py`): `User.email` now `unique=True`; `username` nullable; added
  `oauth_provider` / `oauth_sub`.
- **Frontend** (`templates/index.html` + `static/js/app.js`): `init()` calls `/auth/me`; logged
  out → "Login with Google" gate + disabled composer; logged in → email + Logout. `requireAuth()`
  drops to the gate on any 401 mid-session. Same-origin `fetch`, so the cookie rides automatically.
- **`.env.example`**: created (was missing) — full template incl. the Google + session vars.
- **Tests**: `tests/test_auth.py` (8, via `TestClient` with Google mocked, offline): enforcement
  401s, `DEV_AUTH_BYPASS`, callback creates-user/sets-session, unverified-email 403, logout,
  same-email-same-account. **Suite 98 → 106**, ~1s, no API keys.

**⚠️ Same DB reset still applies** (the Increment 1 `user_id` column + now `User.email unique` /
new oauth columns): `rm -f app.db && rm -rf faiss_index/` before first run, then re-ingest.

**Manual Google Cloud setup (free; required to actually log in):**
1. console.cloud.google.com → APIs & Services → **OAuth consent screen** → External; add your
   own Google account under **Test users**.
2. **Credentials → Create OAuth client ID → Web application.** Authorized redirect URI:
   `http://localhost:8000/auth/callback`.
3. Put the client ID/secret + a real `SESSION_SECRET_KEY`
   (`python -c "import secrets; print(secrets.token_hex(32))"`) in `.env`.
4. To run locally *before* doing this, set `DEV_AUTH_BYPASS=true` in `.env` (dev user, no
   login). Required anyway now: with no `SESSION_SECRET_KEY`, the app refuses to start unless
   `DEV_AUTH_BYPASS=true` or `DEBUG=true`.

**Adversarial security review (2026-06-16): 7 findings, all fixed before commit.** A multi-agent
review (4 reviewers × independent verifiers) confirmed and we fixed: (HIGH) session cookie now
gets `Secure` via `SESSION_COOKIE_SECURE`; (HIGH) insecure default secret now fails closed at
startup + `.env.example` ships it empty; (MED) `/documents/stats/overview` **and** `/chat/status`
were unauthenticated aggregate leaks → now `Depends(get_current_user)`; (MED) `DEV_AUTH_BYPASS`
now logs a loud startup warning; (MED) `find_or_create_user` logs on `oauth_sub` mismatch
(email-reuse signal); (MED) the create-create race now catches `IntegrityError` and re-queries;
(MED) `email_verified` check is fail-closed (`if not ...`). Tests in `test_auth.py` cover the
gating + the verified-email gate. Suite **106 → 109**.

**Still global / to gate later** (carried from Increment 1, intentionally): `POST
/documents/{id}/process`, `POST /documents/rebuild-index`, `DELETE /documents/all`.
`/auth/logout` is a GET (CSRF-able logout — low severity; make it POST if hardening). No CSRF
tokens on state-changing POSTs yet. `/chat/status` + `/documents/stats/overview` are auth-gated
but their counts are still instance-global (per-user scoping deferred).

Roughly the order, each its own commit-able unit:

1. **Multi-tenancy refactor**: add a `user_id` foreign key on `LoreDocument`, stamp `user_id`
   into chunk metadata at ingest, thread it through `vector_store_manager._build_filter_function`,
   and enforce on retrieval; scope document list/delete and conversation sessions by user.
   (Not `ChatHistory` — that schema/table was removed/deferred as dead code.)
2. **Auth**: **authlib** Google OAuth + a Starlette `SessionMiddleware` signed-cookie session
   (the cookie rides the existing same-origin `fetch` frontend automatically). Adds
   login/callback/logout + `/auth/me`, and rewires the `get_current_user` seam from item 1.
3. **Postgres + Alembic**: switch `DATABASE_URL` from sqlite to postgres; add migrations.
4. **Dockerfile + Fly.io deploy**: multi-stage build with the embedding model baked in; persistent volume for FAISS; `fly.toml`.
5. **CI**: GitHub Actions running `uv sync` + `pytest` on PR.
6. **Streaming responses**: switch `llm.invoke()` to `llm.astream()` and pipe through FastAPI's `StreamingResponse`.

## OAuth2 options (expanding Phase 2 item 2)

Notes on how hard auth actually is here. Today there's zero auth — every endpoint is public and
the `User` model in `app/database.py` is defined but unwired. OAuth2 is an *authorization*
framework: a trusted provider (Google/GitHub) vouches for identity so we never store passwords.
Standard web flow is the Authorization Code flow (login redirect → provider → callback with a
short-lived code → backend swaps it for an access token → we issue our own session/JWT).

Ranked easiest → most control:

1. **Managed provider (Auth0 / Clerk / Supabase Auth)** — easiest. ~20 lines + a JS widget; they
   own token storage, refresh, social logins, MFA. Free tiers exist. Cost: vendor dependency.
2. **`fastapi-users` with an OAuth2 backend** — *the path already named in Phase 2 item 2.*
   Gives us login UI hooks, `/me`, cookie+JWT sessions, and pluggable OAuth2 providers
   (Google/GitHub) without vendor lock-in. Integrates cleanly with our SQLAlchemy `User` model.
   Best fit given the existing stack.
3. **`authlib` social login (Google/GitHub) by hand** — more wiring than `fastapi-users` but
   maximum transparency; store only `oauth_provider` + `oauth_id` + email/name.
4. **Self-hosted IdP (Keycloak)** — enterprise SSO, heavy ops overhead, overkill here.

**Recommendation (CORRECTED 2026-06-15):** use **option 3 — `authlib` Google OAuth by hand +
a signed-cookie session** (Starlette `SessionMiddleware` + `itsdangerous`), **not**
`fastapi-users`. Reason discovered after this section was first written: our SQLAlchemy stack
is fully *synchronous*, and `fastapi-users`' DB adapter is **async-only** — adopting it would
bolt an async engine (aiosqlite) onto the sync stack or force a sync→async migration. With
Google-only login there are no passwords to manage, so `fastapi-users`' batteries are mostly
unused here. authlib + cookie is the simplest thing that fully works, fits the sync stack, is
free, and rides the existing same-origin `fetch` frontend. Still done *after* the multi-tenancy
refactor (item 1), since auth only matters once data carries a `user_id` to scope against.

Rough effort: ~half to a full day, mostly new files (auth routes + `get_current_user` dependency,
`SessionMiddleware` registration, login/logout UI in `templates/index.html` + `static/js/app.js`,
Google OAuth2 app credentials in a new `.env.example`).

## Provider / eval environment state (updated 2026-06-15)

What the next session needs to know before running anything live:

- **Google (Gemini, first in fallback order)**: key works, but free tier is **5 requests/min
  AND 20 requests/day per model**; resets daily. 429s name the exact quota violated.
  `DEFAULT_GEMINI_MODEL=gemini-2.5-flash` is a **thinking model** — its reasoning eats the
  output-token budget, so the chapter-labelling call needs `LLM_CHAPTER_DETECTION_MAX_TOKENS`
  ≈16000 (it scored 1.00 on Dune at 16000; truncates below). `gemini-1.5-flash` (404) and
  `gemini-2.0-flash` (empty response) are NOT available on this key — no cheap non-thinking
  Gemini option here yet.
- **OpenAI (GitHub Models PAT, `OPENAI_BASE_URL=https://models.inference.ai.azure.com`,
  gpt-4o-mini)**: reachable, but **quality drifted** — the hybrid detector scored 1.00 via
  the PAT on 2026-06-12 and only **0.17 on 2026-06-15** (stable across 3 runs; it over-omits
  sections). Don't assume a past PAT score still holds; re-measure. Still fine for short Q&A.
  The fine-grained PAT needed the **Models: Read-only** account permission (added 2026-06-12).
  Hard **8k-token request cap** (413 above it): fits the hybrid detector prompt (~5k tokens)
  but rejects the LLM-solo prompt (~70k on a real book).
- **Anthropic**: dead — `400: credit balance too low`. Needs credits to revive.
- `tests/eval/results/chapter_llm.json` holds the only valid Gemini-measured LLM-solo
  real-book baseline (Dune boundary_f1 0.18). A rerun while Gemini is quota-dead writes
  zeros for that row (LLM-solo can't fit through the PAT) — **don't let a provider-failure
  artifact overwrite it** (restore with `git checkout d659dd7 -- tests/eval/results/chapter_llm.json`).
- The Dune fixture (`tests/eval/fixtures/dune_chapters.json`) is gitignored (embeds the
  book text); regenerate from `Books/` with `uv run python -m tests.eval.build_dune_fixture`.

## Run commands you'll want again

```bash
# Tests
uv run pytest tests/

# Retrieval eval
uv run python -m tests.eval.run_eval --rebuild

# Chapter-detection eval, live (env var overrides .env; --sleep paces free-tier limits)
MAX_TOKENS=8000 uv run python -m tests.eval.run_chapter_eval --sleep 15

# Chapter-detection eval, deterministic harness check (no API keys; keep live artifacts safe)
uv run python -m tests.eval.run_chapter_eval --mock-fixture --results-dir /tmp/chapter_eval_mock

# Rebuild the gitignored real-book fixture from Books/
uv run python -m tests.eval.build_dune_fixture

# Smoke test imports
uv run python -c "from main import app; print('OK')"

# Run dev server
uv run uvicorn main:app --reload
```

For WSL-from-Windows, prefix any of these with `wsl.exe -d Ubuntu -- bash -lc '...'`.
