"""Tiny retrieval eval for the current RAG stack.

Builds a FAISS index from a book (default: first .epub in Books/), runs the
hand-authored Q/A pairs from dune_qa.jsonl, and computes recall@5 with
keyword-containment as the relevance signal.

Usage (from project root):
    uv run python -m tests.eval.run_eval
    uv run python -m tests.eval.run_eval --book Books/dune.pdf --rebuild

The FAISS index is cached at tests/eval/.cache/ so re-runs are fast.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOKS_DIR = PROJECT_ROOT / "Books"
DEFAULT_QA = PROJECT_ROOT / "tests" / "eval" / "dune_qa.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "tests" / "eval" / "results" / "baseline.json"
DEFAULT_CACHE = PROJECT_ROOT / "tests" / "eval" / ".cache" / "faiss_index"


# ---------- Loading ----------

def find_default_book() -> Path:
    epubs = sorted(DEFAULT_BOOKS_DIR.glob("*.epub"))
    if not epubs:
        raise SystemExit(f"No .epub files found in {DEFAULT_BOOKS_DIR}. Pass --book.")
    return epubs[0]


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


# ---------- EPUB extraction (mirrors documents_routes._extract_epub_content) ----------

def extract_epub_text(path: Path) -> str:
    """Extract text from EPUB with `=== title ===` chapter separators.

    Mirrors the production path in app/api/documents_routes.py:_extract_epub_content
    so the eval reflects what users actually get when they upload an EPUB.
    """
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    spine_ids = [item[0] for item in book.spine]
    chapters = []
    chapter_num = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if item.get_id() not in spine_ids:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        for elem in soup(["script", "style", "nav"]):
            elem.decompose()

        title = None
        for tag in ["h1", "h2"]:
            elem = soup.find(tag)
            if elem:
                t = elem.get_text(strip=True)
                if t and 2 < len(t) < 200:
                    title = t
                    break

        text_parts = []
        for tag in ["p", "div", "blockquote", "li"]:
            for element in soup.find_all(tag):
                text = element.get_text(" ", strip=True)
                if text and len(text) > 5:
                    text_parts.append(text)

        chapter_text = "\n\n".join(text_parts)
        if len(chapter_text.strip()) < 50:
            continue

        chapter_num += 1
        chapter_title = title or f"Section {chapter_num}"
        chapters.append(f"=== {chapter_title} ===\n\n{chapter_text}")

    return "\n\n".join(chapters)


def extract_pdf_text(path: Path) -> str:
    from langchain_community.document_loaders import PyPDFLoader
    pages = PyPDFLoader(str(path)).load()
    return "\n\n".join(p.page_content for p in pages)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".epub":
        return extract_epub_text(path)
    if ext == ".pdf":
        return extract_pdf_text(path)
    raise SystemExit(f"Unsupported book extension: {ext}")


# ---------- Chunking (uses DocumentManager helpers via __new__) ----------

def chunk_content(content: str, document_title: str):
    from app.services.document_manager import DocumentManager
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    dm = DocumentManager.__new__(DocumentManager)
    dm.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    base_metadata = {
        "document_id": 1,
        "document_title": document_title,
        "source_type": "eval",
    }
    chapters = dm._detect_chapters_in_content(content)
    if len(chapters) >= 2:
        return dm._chunk_with_chapters(content, chapters, base_metadata)
    return dm._chunk_flat(content, base_metadata)


# ---------- Index build / load ----------

def build_or_load_index(book_path: Optional[Path], cache_dir: Path, rebuild: bool):
    from app.services.vector_store_manager import VectorStoreManager

    if rebuild and cache_dir.exists():
        import shutil
        logger.info("Removing cache at %s", cache_dir)
        shutil.rmtree(cache_dir)

    vsm = VectorStoreManager(persist_path=str(cache_dir))

    if vsm.vector_store is not None:
        # Reuse a cached index — e.g. `--cache-dir faiss_index` to evaluate against
        # the production (hybrid-chunked) index, where chapter numbers match the app.
        logger.info("Reusing cached index at %s", cache_dir)
        return vsm

    if book_path is None:
        book_path = find_default_book()
    if not book_path.exists():
        raise SystemExit(f"Book not found: {book_path}")

    logger.info("Building index from %s", book_path.name)
    content = extract_text(book_path)
    logger.info("Extracted %d characters", len(content))

    chunks = chunk_content(content, document_title=book_path.stem)
    logger.info("Chunked into %d documents", len(chunks))

    if not vsm.add_documents(chunks):
        raise SystemExit("Failed to build vector store")

    return vsm


# ---------- Eval ----------

def evaluate(vsm, qa_pairs: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    results = []
    for pair in qa_pairs:
        # Optional per-question spoiler filters — lets a row reproduce the
        # spoiler-mode recall miss (references excluded) where it actually occurs.
        retrieved = vsm.search_with_scores(
            pair["question"],
            k=k,
            max_chapter=pair.get("max_chapter"),
            include_reference=pair.get("include_reference", False),
        )
        keywords = [kw.lower() for kw in pair["expected_chunks_contain_keywords"]]

        matched = set()
        retrieved_chapters = set()
        for doc, _ in retrieved:
            content_lower = doc.page_content.lower()
            for kw in keywords:
                if kw in content_lower:
                    matched.add(kw)
            ch = doc.metadata.get("chapter_number")
            if ch is not None:
                retrieved_chapters.add(ch)

        # chapter_hit is the honest signal for answers written ALLUSIVELY (e.g. a
        # death scene that never says "died") where keyword-containment is fooled:
        # did we retrieve a chunk from the chapter that actually holds the answer?
        expected_chapter = pair.get("expected_chapter")
        chapter_hit = (
            expected_chapter in retrieved_chapters if expected_chapter is not None else None
        )

        results.append({
            "question": pair["question"],
            "expected_keywords": pair["expected_chunks_contain_keywords"],
            "matched_keywords": sorted(matched),
            "hit": len(matched) > 0,
            "expected_chapter": expected_chapter,
            "chapter_hit": chapter_hit,
            "retrieved_chapters": sorted(retrieved_chapters),
            "retrieved_count": len(retrieved),
        })
    return results


def summarize(results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    summary = {
        "metric": f"recall@{k}",
        "hits": hits,
        "total": total,
        "score": hits / total if total else 0.0,
    }
    # Secondary metric over only the rows that pin an expected_chapter.
    chaptered = [r for r in results if r.get("chapter_hit") is not None]
    if chaptered:
        ch_hits = sum(1 for r in chaptered if r["chapter_hit"])
        summary["chapter_recall"] = {
            "hits": ch_hits,
            "total": len(chaptered),
            "score": ch_hits / len(chaptered),
        }
    return summary


# ---------- Entry ----------

def main():
    parser = argparse.ArgumentParser(description="Recall@5 eval for current RAG stack")
    parser.add_argument("--book", type=Path, default=None, help="EPUB or PDF (default: first .epub in Books/)")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rebuild", action="store_true", help="Discard cached index and rebuild")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa)
    logger.info("Loaded %d Q/A pairs", len(qa_pairs))

    vsm = build_or_load_index(args.book, args.cache_dir, args.rebuild)
    results = evaluate(vsm, qa_pairs, k=args.k)
    summary = summarize(results, k=args.k)

    output = {
        "timestamp": datetime.now().isoformat(),
        "book": args.book.name if args.book else args.cache_dir.name,
        "stack": {
            "embeddings": "all-MiniLM-L6-v2",
            "chunker": "RecursiveCharacterTextSplitter(1000/200)",
            "reranker": vsm.reranker.model_name if vsm.reranker else None,
        },
        "summary": summary,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nrecall@{args.k}: {summary['score']:.2f} ({summary['hits']}/{summary['total']})")
    if "chapter_recall" in summary:
        cr = summary["chapter_recall"]
        print(f"chapter-recall@{args.k}: {cr['score']:.2f} ({cr['hits']}/{cr['total']})")
        for r in results:
            if r.get("chapter_hit") is False:
                print(
                    f"  MISS  ch{r['expected_chapter']} not retrieved "
                    f"(got {r['retrieved_chapters']}): {r['question']}"
                )
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
