"""
Enhanced RAG Service - Focuses on RAG queries and answer generation.
Supports simplified spoiler filtering with optional reference material.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.config import settings
from app.services.document_manager import DocumentManager
from app.services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """
    Enhanced RAG service focused on query processing and answer generation.
    """

    def __init__(self):
        logger.info("Initializing Enhanced RAG Service...")

        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager()

        # Initialize document manager
        self.document_manager = DocumentManager(self.vector_store_manager)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Conversational features
        self.context_aware_rag = None
        self._setup_conversational_rag()

        logger.info("Enhanced RAG Service initialized")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on available API keys."""

        timeout = settings.LLM_REQUEST_TIMEOUT
        max_retries = settings.LLM_MAX_RETRIES

        if settings.OPENAI_API_KEY and settings.DEFAULT_OPENAI_MODEL:
            logger.info("Using OpenAI: %s", settings.DEFAULT_OPENAI_MODEL)
            return ChatOpenAI(
                model_name=settings.DEFAULT_OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=settings.MAX_TOKENS,
                request_timeout=timeout,
                max_retries=max_retries,
            )
        if settings.ANTHROPIC_API_KEY and settings.DEFAULT_CLAUDE_MODEL:
            logger.info("Using Claude: %s", settings.DEFAULT_CLAUDE_MODEL)
            return ChatAnthropic(
                model=settings.DEFAULT_CLAUDE_MODEL,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.3,
                max_tokens=settings.MAX_TOKENS,
                default_request_timeout=timeout,
                max_retries=max_retries,
            )
        if settings.GOOGLE_API_KEY and settings.DEFAULT_GEMINI_MODEL:
            logger.info("Using Google Gemini: %s", settings.DEFAULT_GEMINI_MODEL)
            # Note: ChatGoogleGenerativeAI exposes timeout/retries via the
            # underlying transport; LangChain's wrapper accepts max_retries
            # and a `timeout` kwarg in newer releases. We pass what we can
            # and let langchain ignore unknowns rather than break here.
            return ChatGoogleGenerativeAI(
                model=settings.DEFAULT_GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3,
                max_tokens=settings.MAX_TOKENS,
                timeout=timeout,
                max_retries=max_retries,
            )
        logger.warning("No LLM configured - check your API keys")
        return None

    def _setup_conversational_rag(self):
        """Initialize conversational RAG system."""
        if self.llm:
            from app.services.conversational_memory import initialize_context_aware_rag
            self.context_aware_rag = initialize_context_aware_rag(self)
        else:
            logger.warning("Conversational features not available (no LLM)")

    # Prompt for simple Q&A. Allows synthesis across retrieved chunks but
    # strictly limits the model to provided context.
    _ANSWER_PROMPT = PromptTemplate(
        template=(
            "You are an expert Reading Companion and Lorekeeper.\n"
            "Your goal is to help the user understand the world, remember characters, "
            "and track plotlines.\n\n"
            "Context from the book/documents:\n{context}\n\n"
            "User's Question: {question}\n\n"
            "Instructions:\n"
            "1. **Role**: Act as a helpful guide. If asked \"Who is X?\", provide their "
            "identity, allegiance, and key relationships based on the context.\n"
            "2. **Terminology**: If unique or technical terms appear in the context, "
            "define them briefly if relevant to the answer.\n"
            "3. **Synthesis Allowed**: Base your answer *only* on the provided context, "
            "but you may synthesize details from multiple sections to form a complete "
            "answer. Do not use outside knowledge.\n"
            "4. **Spoilers**: Answer the specific question asked. Do not reveal major "
            "future plot twists unless explicitly asked.\n"
            "5. **Clarity**: Be precise with spelling and relationships.\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    async def ask_question(
        self,
        question: str,
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None,
        include_reference: bool = False,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Returns a dict with answer, sources (with real similarity scores),
        and a confidence aggregated from those scores.
        """

        if not question.strip():
            return {"error": "Question cannot be empty"}

        if not self.vector_store_manager.vector_store:
            return {
                "answer": (
                    "I don't have any documents to search through. "
                    "Please upload and process some documents first!"
                ),
                "sources": [],
                "confidence": None,
                "chunks_used": 0,
            }

        if not self.llm:
            return {"error": "No language model configured. Please check your API keys."}

        retrieval_k = k if k is not None else settings.RETRIEVAL_K

        # Log retrieval scope (filters + k) so we can audit per-query behavior.
        filter_info = []
        if document_id is not None:
            filter_info.append(f"document {document_id}")
        if max_chapter is not None:
            filter_info.append(f"chapters 1-{max_chapter}")
            if include_reference:
                filter_info.append("+ reference material")
        scope = ", ".join(filter_info) if filter_info else "all documents"
        logger.info("RAG retrieval (k=%d, scope=%s) for question: %r", retrieval_k, scope, question)

        try:
            docs_with_scores = self.vector_store_manager.search_with_scores(
                question,
                k=retrieval_k,
                document_id=document_id,
                max_chapter=max_chapter,
                include_reference=include_reference,
            )

            if not docs_with_scores:
                return {
                    "answer": (
                        "I couldn't find any relevant passages for that question "
                        "given the current filters."
                    ),
                    "sources": [],
                    "confidence": None,
                    "chunks_used": 0,
                    "spoiler_filter_active": max_chapter is not None,
                    "max_chapter": max_chapter,
                    "include_reference": include_reference,
                }

            answer = self._invoke_llm_with_context(question, docs_with_scores)
            sources = self._format_sources(docs_with_scores)
            confidence = self._aggregate_confidence([s["similarity_score"] for s in sources])

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(sources),
                "spoiler_filter_active": max_chapter is not None,
                "max_chapter": max_chapter,
                "include_reference": include_reference,
            }

        except (ValueError, RuntimeError) as e:
            logger.exception("Error generating answer")
            return {"error": str(e)}

    def _invoke_llm_with_context(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> str:
        """Format retrieved chunks into the prompt and call the LLM."""
        context = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)
        prompt_text = self._ANSWER_PROMPT.format(context=context, question=question)

        response = self.llm.invoke(prompt_text)
        # ChatModels return AIMessage; older string-LLM compat returns str.
        return getattr(response, "content", None) or str(response)

    def _format_sources(
        self,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[Dict[str, Any]]:
        """Build the API source list with real cosine-style similarity scores."""
        sources: List[Dict[str, Any]] = []
        for i, (doc, raw_score) in enumerate(docs_with_scores):
            source_info: Dict[str, Any] = {
                "document_title": doc.metadata.get("document_title", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", i),
                "similarity_score": VectorStoreManager.normalize_score(raw_score),
            }

            chapter_num = doc.metadata.get("chapter_number")
            chapter_title = doc.metadata.get("chapter_title")
            is_ref = doc.metadata.get("is_reference", False)

            if chapter_title:
                source_info["chapter_title"] = chapter_title
            if chapter_num:
                source_info["chapter_number"] = chapter_num
            if is_ref:
                source_info["is_reference"] = True

            sources.append(source_info)
        return sources

    @staticmethod
    def _aggregate_confidence(scores: List[Optional[float]]) -> Optional[float]:
        """Average non-null similarity scores into a single confidence value."""
        valid = [s for s in scores if s is not None]
        if not valid:
            return None
        return sum(valid) / len(valid)

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""

        stats = self.document_manager.get_stats()

        return {
            "documents_loaded": stats['processed_documents'],
            "total_chunks": stats['total_chunks'],
            "deleted_documents": stats['deleted_documents'],
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "FAISS",
            "llm_available": self.llm is not None,
            "conversational_available": self.context_aware_rag is not None,
            "status": "ready" if stats['total_chunks'] > 0 and self.llm else "not_ready",
            "should_rebuild": stats['should_rebuild']
        }


# Global service instance
enhanced_rag_service = EnhancedRAGService()