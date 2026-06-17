# Retrieval of implicit / allusive content — problem & options (research notes)

_Created 2026-06-17. Reference notes, not a committed plan. Captures why semantic retrieval
misses obliquely-written facts and how the field addresses it, so we can revisit retrieval
quality from an informed position._

## The concrete trigger

Spoiler-mode query (refs off): **"What happens to Baron Vladimir Harkonnen at the end? Does he die?"**
The answer is in narrative Ch 47, written allusively:
> "'You've met the Atreides gom jabbar.' She … dropped a dark needle … The Baron fell back."

It never says *die / death / Vladimir / Harkonnen*. Measured against the production index
(1,573 chunks, MiniLM embeddings + bge-reranker):

- The death chunk ranks **369 / 1,573** in dense retrieval (cosine sim 0.286).
- The *best* Ch 47 chunk ranks only **~55** ("…this old Harkonnen beast … my father is dead",
  sim 0.398) — below the rerank pool of 30, so the cross-encoder never even sees it.
- Top results are all "Baron scheming/plotting" chunks (ch 8, 21, 26, 28, 35) — they share
  vocabulary with the query and correctly out-score the answer.

## Why this is inherent, not a bug

Dense retrieval ranks by surface semantic/vocabulary proximity between the question's phrasing
and the chunk's phrasing. An obliquely-written event lives in a different neighborhood than the
literal question, while passages that *talk about* the subject in the query's own words dominate.
The reranker can't rescue it: it only reorders what it's given (and the best Ch 47 chunk isn't in
the pool), and it's still a text-vs-text matcher with the same blind spot. This is the
**"lexical / semantic gap,"** and in fiction it's an active research area (implicit single-mention
facts, multi-hop fragmentation, out-of-order chunks) — see NovelHopQA, FictionRAG below.

**The eval harness is the decision instrument.** `tests/eval/run_eval.py` measures recall@5
(keyword containment) plus **chapter-recall** (did retrieval surface the chapter that holds the
answer — the honest signal for allusive answers). Any future fix must improve chapter-recall on
*several* allusive questions **without** regressing recall@5 (currently 0.90). One example (the
Baron) is gameable; add 2–3 more allusive rows before trusting any change.

## The fix families (every approach does one of two things)

### Approach A — add a *second way to search* (additive, lower-risk; leaves working dense retrieval intact)

| Technique | What it is (plain) | Fixes the Baron? | Cost / risk |
|---|---|---|---|
| **Hybrid keyword + meaning (BM25 + dense, RRF fusion)** | Run exact-word matching alongside meaning-search, merge the lists. The near-universal first move. | No (death chunk shares no words with the query) — but would likely fix the *Emperor* rare-name miss. | Free, no LLM calls, additive. New dep `rank_bm25`. |
| **HyDE — "answer first, then search"** | LLM writes a *hypothetical* answer passage; search with that. Works even if the fake answer is wrong, because its *shape* lands near a real death scene. | Partially — but for a famous book it leans on the model's training knowledge of Dune (conflicts with the honesty rule) and adds a per-query LLM call. | Per-query LLM latency; outside-knowledge tension. |
| **ColBERT / late interaction** | Finer-grained matcher: compares word-by-word instead of one fingerprint per passage. | Helps generally; not a targeted cure. | Different, much larger index — real engineering lift. |

### Approach B — make the implied fact *explicit in the index* (more powerful for this exact problem; costs LLM work at book-load time)

| Technique | What it is (plain) | Fixes the Baron? | Cost / risk |
|---|---|---|---|
| **Anthropic Contextual Retrieval** | Prepend a one-sentence context to every chunk before indexing ("This is the scene where the Baron is killed…"). Cuts retrieval failures ~49–67%. | Yes, directly. | ~1 LLM call/chunk, but with prompt caching ~**$12 / 1,000 pages**. Needs a working paid key (Anthropic credits currently empty). |
| **RAPTOR — tree of summaries** | Recursively cluster + summarize passages into a tree; high-level summary nodes state "the Baron dies." +20% on a hard long-doc QA benchmark. | Yes. | Many LLM calls at ingest (heavy). |
| **GraphRAG (Microsoft)** | Build a character/event graph; "what happens to X?" fans out from the X node to connected events incl. the death, regardless of wording. Purpose-built for this question type. | Yes — best-targeted. | Heaviest ingest (entity/graph extraction + community summaries). |
| **Per-chapter summary nodes** _(our lightweight take — a one-level RAPTOR)_ | At ingest, write one short summary per chapter and index it carrying that chapter's number; the Ch 47 summary states the death plainly. | Yes, for "what happens / fate" questions. | ~48 LLM calls/book (feasible on free tier); spoiler-safe via `chapter_number`; risk = mis-tagged summary leaking a spoiler (needs a test). |

## Honest bottom line for this project

- **No free lunch.** Robustly fixing implicit-fact retrieval costs *either* a second search
  index (A) *or* LLM work at book-load time (B). Everyone pays one of those.
- **Lowest-risk additive move:** hybrid keyword+meaning search. Won't fix the Baron, but it's
  safe and helps other misses (e.g. the Emperor).
- **Best-targeted at "what happens to character X":** GraphRAG / RAPTOR — both heavy.
  Contextual Retrieval is the most affordable of the index-enrichment family (caching), but needs
  a paid key. The per-chapter-summary idea is the budget version of the same principle.
- **Accepting it as a documented limitation is legitimate.** The answer prompt's honesty rule
  already prevents the app from inventing an answer when retrieval misses; many production systems
  document such gaps rather than chase every one.

## Sources

- Hybrid lexical+semantic / RRF: https://www.keyvalue.systems/blog/hybrid-rag-architecture/
- Lexical-mismatch / synonym-aware RAG: https://live.paloaltonetworks.com/t5/engineering-blogs/bridging-the-language-gap-our-journey-to-a-synonym-aware-rag/ba-p/1236616
- Query expansion: https://medium.com/@sahin.samia/query-expansion-in-enhancing-retrieval-augmented-generation-rag-d41153317383
- HyDE (Precise Zero-Shot Dense Retrieval without Relevance Labels): https://arxiv.org/abs/2212.10496
- ColBERT / late interaction overview: https://weaviate.io/blog/late-interaction-overview
- Anthropic Contextual Retrieval guide: https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide
- RAPTOR (paper): https://arxiv.org/pdf/2401.18059  ·  code: https://github.com/parthsarthi03/raptor
- GraphRAG (Microsoft docs): https://microsoft.github.io/graphrag/  ·  "From Local to Global" paper: https://arxiv.org/pdf/2404.16130
- NovelHopQA (multi-hop reasoning in long narratives): https://arxiv.org/pdf/2506.02000
- FictionRAG (long-narrative framework): https://doi.org/10.3390/a19050383
