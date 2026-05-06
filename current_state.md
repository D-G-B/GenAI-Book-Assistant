# Current State

A snapshot of where the project sits at the end of the Phase 1 hardening pass.

## Branch

`cleanup-and-improvement` — 8 commits ahead of `main`. Merge candidate; everything verified locally.

```
ef2cc94 Phase 1A:   gate uvicorn reload on DEBUG, sync uv.lock, ignore .claude/
9ee7808 Phase 1B.1: project-wide logging, replace print() with loggers, tighten excepts
55dc8f7 Phase 1B.2: pytest infra + unit tests for spoiler filter, chapter extraction, score normalization
1c3cd54 Phase 1B.3: tiny retrieval eval + Dune baseline (recall@5 = 0.80)
210a9e7 Phase 1C.1: replace ConversationalRetrievalChain with direct flow, populate similarity scores
6608e3a Phase 1C.2: cross-encoder reranker (BAAI/bge-reranker-base), recall@5 0.80 -> 0.90
b18e744 Phase 1C.3: batch add_documents (one embed call per upload, with per-chunk fallback)
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

## Run commands you'll want again

```bash
# Tests
uv run pytest tests/

# Retrieval eval
uv run python -m tests.eval.run_eval --rebuild

# Smoke test imports
uv run python -c "from main import app; print('OK')"

# Run dev server
uv run uvicorn main:app --reload
```

For WSL-from-Windows, prefix any of these with `wsl.exe -d Ubuntu -- bash -lc '...'`.
