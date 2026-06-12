# Current State

A snapshot of where the project sits at the end of the Phase 1 hardening pass.

## Branch

`claude/upbeat-lovelace-a4qv7` — 9 commits ahead of `main`, all pushed to origin. (The
Phase 1 work formerly tracked here on `cleanup-and-improvement` is already in `main`.)
Latest session (2026-06-12):

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

- **Titled-only chapters silently break spoiler protection.** `_detect_chapters_in_content`
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

  **Next session: wire the hybrid into ingest.** Decided design (2026-06-12):

  - In `DocumentManager._process_and_chunk` (`document_manager.py:365`): call
    `detect_chapters_hybrid(content)` first; if it returns >= 2 chapters, chunk with that;
    otherwise fall through to the existing `_detect_chapters_in_content` regex (current
    behavior preserved — the hybrid already returns `[]` on any LLM failure precisely so
    this fallback is trivial).
  - **No "zero-cost tier"** (skipping the LLM when regex finds genuine `Chapter N`
    headings was considered and rejected). Why: finding chapter headings is not the same
    as understanding front/back matter. Traced through `_chunk_with_chapters` +
    `vector_store_manager._build_filter_function`: (a) Foreword/Preface/Prologue/
    Introduction have no regex pattern, so they land in the "Frontmatter" bucket
    (`chapter_number=None`) and are blocked whenever spoiler protection is on — safe but
    invisible; (b) **sequel-preview sections headed "Chapter 1" get chapter_number=1 and
    leak through every max_chapter setting** — a real spoiler hole; (c) part-relative
    numbering ("Part II, Chapter 1") collides the same way. All three are semantic calls
    the hybrid's LLM pass makes correctly; at ~$0.002/book once at ingest the saving is
    not worth the leak.
  - Integration prerequisites: `invoke_with_fallback` caps output at `settings.MAX_TOKENS`
    (1000 in .env today) — real-book labelling needs ~2-3k output tokens, so raise
    MAX_TOKENS in .env or add a per-call override; `_process_and_chunk` is async while the
    LLM call blocks (acceptable, or wrap in `run_in_executor`); consider a
    `LLM_CHAPTER_DETECTION_ENABLED` settings flag for opt-out.
  - Optional, low priority, independent: harden the regex fallback by adding
    Foreword/Preface/Prologue/Introduction to `reference_patterns`, and note the
    pattern-priority quirk (a titled `=== ... ===` match within 20 chars beats reference
    classification — why Dune's appendices weren't flagged).

The following were found in a codebase sanity sweep (2026-06); none are fixed yet:

- **CRITICAL — `should_rebuild()` ratio formula is wrong** (`vector_store_manager.py:370`).
  `deleted_ratio = deleted_count / (deleted_count + 10)` doesn't use `total_chunks` at all, so
  the rebuild trigger almost never fires. Soft-deleted chunks accumulate in the FAISS index
  indefinitely, degrading retrieval and wasting disk. Fix: `deleted_count / total_chunks`.
- **`validate_api_keys()` doesn't validate model names** (`config.py:40`). A key without its
  matching `DEFAULT_*_MODEL` boots cleanly, then every chat request fails at query time.
- **No file size limit on upload** (`documents_routes.py:82`). `await file.read()` loads the
  whole file into memory before any check; a huge upload can hang/OOM the server.
- **Unsupported file types stored as junk content** (`documents_routes.py:112`).
  `"[Unsupported file type: foo.exe]"` is saved as real document content and later fails
  downstream during chunking. Should 400 at upload instead.
- **`JSONLoader(text_content=False)`** (`advanced_document_loaders.py:392`) returns metadata
  rather than text, likely silently breaking `.json` ingestion. Should be `text_content=True`.
- **No input bounds on chat** (`chat_routes.py:33`). Question length is unbounded (token
  waste) and `max_chapter` accepts negative/absurd values.
- **Nullable DB columns assumed non-null** (`database.py:54`). `content` and `source_type`
  are nullable but the ingest path dereferences them without checks. `doc_metadata` column is
  defined but never populated.
- **Startup reprocessing swallows failure reasons** (`main.py:34`). When `add_document()`
  returns False at boot, logs say nothing about why.
- **Unused `BaseLoader` import** (`advanced_document_loaders.py:23`) — dead code.

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
- No `user_id` on documents, conversations, or vector store metadata.
- No login. Anyone hitting the API sees everyone's books.

### Deploy
- No Dockerfile.
- No CI (no GitHub Actions, no automated test run on PR).
- SQLite-only (`./rag_assistant.db`); no migrations (no Alembic).
- FAISS is in-process and on-disk; doesn't scale past one container.

### Robustness
- No rate limiting.
- No request size limits beyond FastAPI defaults.
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

Roughly the order I'd attack things, but each is its own commit-able unit:

1. **Multi-tenancy refactor**: add `user_id` foreign key on `LoreDocument`, `ChatHistory`, conversation sessions; thread `user_id` through `vector_store_manager` filter; enforce on retrieval.
2. **Auth**: `fastapi-users` with cookie sessions + JWT. Adds login UI and a `/me` endpoint.
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

**Recommendation:** stick with the Phase 2 plan — `fastapi-users` + a Google OAuth2 backend.
It should be done *after* the multi-tenancy refactor (item 1), since auth is only useful once
documents/conversations carry a `user_id` to scope against.

Rough effort: ~half to a full day, mostly new files (auth routes + `get_current_user` dependency,
`SessionMiddleware` registration, login/logout UI in `templates/index.html` + `static/js/app.js`,
Google OAuth2 app credentials in a new `.env.example`).

## Provider / eval environment state (as of 2026-06-12)

What the next session needs to know before running anything live:

- **Google (Gemini, first in fallback order)**: key works, but free tier is **5 requests/min
  AND 20 requests/day per model** — the daily quota was exhausted on 2026-06-12; it resets
  daily. 429s name the exact quota violated.
- **OpenAI (GitHub Models PAT, `OPENAI_BASE_URL=https://models.inference.ai.azure.com`,
  gpt-4o-mini)**: working — the fine-grained PAT needed the **Models: Read-only** account
  permission (added 2026-06-12). Hard **8k-token request cap** (413 above it): fits the
  hybrid detector prompt (~5k tokens) but rejects the LLM-solo prompt (~70k on a real book).
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
