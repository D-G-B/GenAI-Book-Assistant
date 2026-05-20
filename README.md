# RAG Reading Partner

A FastAPI + LangChain RAG app for chatting with books — PDF, EPUB, and text — with **chapter-aware spoiler filtering** so you can ask questions about chapters you've already read without getting plot points from later in the book.

> Status: Phase 1 hardening complete on the `cleanup-and-improvement` branch. See [current_state.md](./current_state.md) for the full state-of-the-app and roadmap.

## Highlights

- **Chapter-aware spoiler filtering** — set `max_chapter=N` and only chunks with `chapter_number ≤ N` are retrieved. Reference material (glossary, appendices) is opt-in via `include_reference=True`.
- **Multi-format ingestion** — `.pdf`, `.epub`, `.docx`, `.txt`, `.md`, `.csv`, `.json`. Custom EPUB loader builds chapter metadata from the spine.
- **Two chat modes** — simple Q&A and a conversational mode with session memory and pronoun resolution.
- **Cross-encoder reranker** — `BAAI/bge-reranker-base` reorders the top-N FAISS candidates to lift recall on proper-noun queries (recall@5 0.80 → 0.90 on the Dune eval).
- **Real similarity scores** — both Q&A paths return cosine-similarity scores per source chunk (no placeholders).
- **Multi-LLM** — OpenAI, Anthropic Claude, or Google Gemini, picked from whichever API key is set.
- **FAISS persistence** — index lives in `./faiss_index/` and survives restarts.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- (Optional) CUDA GPU — sentence-transformers and the reranker pick it up automatically

### Install

```bash
git clone <repo-url>
cd GenAI-Book-Assistant
uv sync
```

### Configure

Create `.env` in the project root:

```env
# At least one API key is required
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...

# Pick one default model per provider you've enabled
DEFAULT_OPENAI_MODEL=gpt-4o-mini
DEFAULT_CLAUDE_MODEL=claude-3-5-haiku-latest
DEFAULT_GEMINI_MODEL=gemini-2.0-flash

# Optional
DEBUG=False
LOG_LEVEL=INFO
MAX_TOKENS=1000

# Retrieval
RETRIEVAL_K=8
LLM_REQUEST_TIMEOUT=30
LLM_MAX_RETRIES=2

# Reranker (optional; recall@5 boost on proper-noun queries)
RERANKER_ENABLED=True
RERANKER_MODEL=BAAI/bge-reranker-base
RERANK_POOL_SIZE=30
```

### Run

```bash
uv run uvicorn main:app --reload   # dev (auto-reload on file changes)
# or
uv run python main.py              # plain
```

Open http://localhost:8000.

## Testing

The project ships with a unit test suite for the core retrieval IP and a small retrieval eval against Dune.

### Unit tests (fast, no API keys, ~1s)

```bash
uv run pytest tests/
```

Covers:
- `tests/test_spoiler_filter.py` — table-driven over every combination of `max_chapter`, `include_reference`, soft-delete, and document filter (22 tests, the IP).
- `tests/test_chapter_extraction.py` — arabic / word / roman / uppercase chapter markers, reference classification, ordering (13 tests).
- `tests/test_score_normalization.py` — L2 → cosine math and clamping (6 tests).

The legacy live-server scripts (`test_reading_partner.py`, `test_conversational.py`, `add_complex_document.py`) are skipped under pytest via `tests/conftest.py`. Run them directly when a server is up:

```bash
uv run python tests/test_reading_partner.py
```

### Retrieval eval (recall@5)

A 10-question eval over Dune, with the FAISS index cached at `tests/eval/.cache/`:

```bash
# Drop a Dune .epub or .pdf into Books/ first (the folder is gitignored)
uv run python -m tests.eval.run_eval

# Force a fresh index build
uv run python -m tests.eval.run_eval --rebuild

# Use a specific book
uv run python -m tests.eval.run_eval --book Books/dune.pdf

# Override output path
uv run python -m tests.eval.run_eval --out tests/eval/results/my_run.json
```

Saved results live under `tests/eval/results/`:
- `baseline.json` — MiniLM + char chunking, no reranker (recall@5 = 0.80)
- `with_reranker.json` — same stack + cross-encoder rerank (recall@5 = 0.90)

The eval is purely keyword-containment over the top-5 retrieved chunks; it doesn't call an LLM. Treat it as a guard-rail for retrieval changes, not a quality measure of the final answers.

## Architecture

```
project/
├── app/
│   ├── api/                      # FastAPI routers
│   │   ├── chat_routes.py
│   │   ├── conversational_routes.py
│   │   └── documents_routes.py
│   ├── services/
│   │   ├── enhanced_rag_service.py       # simple Q&A path
│   │   ├── conversational_memory.py      # condense → retrieve → answer (LCEL-style)
│   │   ├── document_manager.py           # ingestion, chunking, chapter detection
│   │   ├── advanced_document_loaders.py  # custom EPUB + format dispatch
│   │   ├── vector_store_manager.py       # FAISS + spoiler filter + reranker
│   │   └── reranker.py                   # cross-encoder, lazy-loaded
│   ├── schemas/                  # Pydantic models
│   ├── database.py
│   └── config.py
├── tests/
│   ├── test_spoiler_filter.py
│   ├── test_chapter_extraction.py
│   ├── test_score_normalization.py
│   └── eval/
│       ├── dune_qa.jsonl
│       ├── run_eval.py
│       └── results/
├── static/                       # JS, CSS
├── templates/                    # Jinja2
└── main.py
```

## API

### Documents

- `POST /api/v1/documents/upload-file` — multipart upload (`.pdf`, `.epub`, `.docx`, `.txt`, `.md`, `.csv`, `.json`)
- `GET /api/v1/documents/list?include_deleted=false`
- `GET /api/v1/documents/{id}/status`
- `DELETE /api/v1/documents/{id}` — soft delete
- `DELETE /api/v1/documents/all` — hard delete everything
- `POST /api/v1/documents/rebuild-index`

### Chat

- `POST /api/v1/chat/ask` — simple Q&A; takes `document_id`, `max_chapter`, `include_reference`, `k` query params
- `POST /api/v1/conversation/ask` — conversational; same filters plus `session_id`
- `GET /api/v1/conversation/history/{session_id}`
- `DELETE /api/v1/conversation/session/{session_id}`
- `GET /api/v1/conversation/sessions`
- `GET /health`

## Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `DEBUG` | `False` | Gates `uvicorn --reload` and verbose logging |
| `LOG_LEVEL` | `INFO` | Standard logging level |
| `MAX_TOKENS` | `1000` | LLM max output tokens |
| `RETRIEVAL_K` | `8` | Chunks returned per query |
| `LLM_REQUEST_TIMEOUT` | `30` | LLM call timeout (seconds) |
| `LLM_MAX_RETRIES` | `2` | LLM retry budget |
| `RERANKER_ENABLED` | `True` | Toggle the cross-encoder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Reranker model id |
| `RERANK_POOL_SIZE` | `30` | Candidates to fetch from FAISS before rerank |

## License

MIT.
