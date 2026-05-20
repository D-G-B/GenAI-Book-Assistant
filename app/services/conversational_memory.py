"""
Conversational memory system for context-aware conversations.
Supports simplified spoiler filtering with optional reference material.

Replaces the deprecated ConversationalRetrievalChain with a direct
condense-then-retrieve-then-answer flow so we can return real similarity
scores for retrieved chunks and control retrieval k / filters per-query.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import PromptTemplate

from app.config import settings
from app.services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class ConversationSession:
    """Represents a single conversation session."""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
        })
        self.last_activity = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this conversation."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'message_count': len(self.messages),
        }


class ConversationMemoryManager:
    """Manages multiple conversation sessions with bounded retention."""

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions

    def get_or_create_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> ConversationSession:
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_session()
            self.sessions[session_id] = ConversationSession(session_id, user_id)
        return self.sessions[session_id]

    def _cleanup_oldest_session(self):
        if not self.sessions:
            return
        oldest = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].last_activity,
        )
        del self.sessions[oldest]


# Rewrite a follow-up question into a standalone query using chat history.
_CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the following conversation about a book or story, and a follow-up "
    "question, rephrase the follow-up question to be a standalone question. "
    "Resolve any pronouns (he, she, it, they, his, her) to the specific "
    "characters, places, or objects mentioned in the chat history.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow Up Question: {question}\n\n"
    "Standalone Question:"
)

_QA_PROMPT = PromptTemplate.from_template(
    "You are an expert Reading Companion and Lorekeeper.\n"
    "Your goal is to answer the user's question based ONLY on the context "
    "provided below.\n\n"
    "Rules:\n"
    "1. If the answer is not in the context, say \"I don't know based on the "
    "available chapters.\"\n"
    "2. Do not make up facts or use outside knowledge.\n"
    "3. Be helpful but concise.\n"
    "4. If the context is from reference material (appendix, glossary), mention that.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Number of recent exchanges to surface when condensing the question.
_CONDENSE_HISTORY_WINDOW = 5


def _format_history(messages: List[Dict[str, Any]], window: int) -> str:
    """Render the last `window` exchanges as a transcript for the condense prompt."""
    recent = messages[-window * 2:] if window else messages
    lines = []
    for msg in recent:
        role = "Human" if msg["role"] == "human" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


class ContextAwareRAG:
    """RAG system with conversational memory and spoiler filtering."""

    def __init__(self, base_rag_service):
        self.base_rag = base_rag_service
        self.memory_manager = ConversationMemoryManager()

    async def ask_with_context(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None,
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None,
        include_reference: bool = False,
    ) -> Dict[str, Any]:
        """Ask a question with conversational context and real similarity scores."""

        if not question.strip():
            return {"error": "Question cannot be empty"}

        if not self.base_rag.llm:
            return {"error": "No language model configured"}

        if not self.base_rag.vector_store_manager.vector_store:
            return {"error": "No documents available"}

        try:
            session = self.memory_manager.get_or_create_session(session_id, user_id)

            filter_info = []
            if document_id is not None:
                filter_info.append(f"doc {document_id}")
            if max_chapter is not None:
                filter_info.append(f"ch 1-{max_chapter}")
                if include_reference:
                    filter_info.append("+ refs")
            if filter_info:
                logger.info(
                    "💬 Conversational search with filters: %s", ", ".join(filter_info)
                )

            logger.info(
                "💬 Conversational question (Session: %s...): %s",
                session_id[:8], question,
            )

            # Record the user turn before retrieval so it shows up in history
            # even if downstream calls fail.
            session.add_message('human', question)

            # 1. Resolve pronouns / context into a standalone search query.
            search_query, condense_result = self._condense_question(
                question, session.messages[:-1]
            )

            # 2. Retrieve with real cosine-style similarity scores.
            docs_with_scores = self.base_rag.vector_store_manager.search_with_scores(
                search_query,
                k=settings.RETRIEVAL_K,
                document_id=document_id,
                max_chapter=max_chapter,
                include_reference=include_reference,
            )

            if not docs_with_scores:
                empty_answer = (
                    "I couldn't find any relevant passages for that question "
                    "given the current filters."
                )
                session.add_message('assistant', empty_answer)
                # condense may still have called the LLM; surface that count.
                return self._build_response(
                    answer=empty_answer,
                    sources=[],
                    confidence=None,
                    session=session,
                    document_id=document_id,
                    max_chapter=max_chapter,
                    include_reference=include_reference,
                    llm_provider=condense_result["provider"] if condense_result else None,
                    llm_calls=condense_result["calls"] if condense_result else 0,
                )

            # 3. Generate the answer from the retrieved chunks.
            context = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)
            answer_result = self._invoke_llm(
                _QA_PROMPT.format(context=context, question=question)
            )
            session.add_message('assistant', answer_result["text"])

            sources = self._format_sources(docs_with_scores)
            valid_scores = [
                s["similarity_score"] for s in sources if s["similarity_score"] is not None
            ]
            confidence = sum(valid_scores) / len(valid_scores) if valid_scores else None

            # Telemetry: sum LLM calls across condense + answer; report the
            # provider that produced the final answer (most informative).
            total_calls = answer_result["calls"]
            if condense_result:
                total_calls += condense_result["calls"]

            return self._build_response(
                answer=answer_result["text"],
                sources=sources,
                confidence=confidence,
                session=session,
                document_id=document_id,
                max_chapter=max_chapter,
                include_reference=include_reference,
                llm_provider=answer_result["provider"],
                llm_calls=total_calls,
            )

        except Exception as e:
            logger.exception("Error in conversational question")
            return {"error": str(e)}

    def _condense_question(
        self, question: str, prior_messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Rewrite the question as a standalone query if there's prior history.

        Returns: (rewritten_query, llm_result) where llm_result is the dict
        returned by invoke_with_fallback (text/provider/calls) when an LLM was
        actually called, or None when there's no history (no LLM call made).
        """
        if not prior_messages:
            return question, None

        history_text = _format_history(prior_messages, window=_CONDENSE_HISTORY_WINDOW)
        result = self._invoke_llm(
            _CONDENSE_PROMPT.format(chat_history=history_text, question=question)
        )
        rewritten = result["text"].strip()
        return (rewritten or question), result

    def _invoke_llm(self, prompt_text: str) -> Dict[str, Any]:
        return self.base_rag.invoke_with_fallback(prompt_text)

    @staticmethod
    def _format_sources(docs_with_scores) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for i, (doc, raw_score) in enumerate(docs_with_scores):
            source_info: Dict[str, Any] = {
                "document_title": doc.metadata.get('document_title', 'Unknown'),
                "chunk_index": doc.metadata.get('chunk_index', i),
                "similarity_score": VectorStoreManager.normalize_score(raw_score),
            }
            chapter_num = doc.metadata.get('chapter_number')
            chapter_title = doc.metadata.get('chapter_title')
            is_ref = doc.metadata.get('is_reference', False)
            if chapter_title:
                source_info['chapter_title'] = chapter_title
            if chapter_num:
                source_info['chapter_number'] = chapter_num
            if is_ref:
                source_info['is_reference'] = True
            sources.append(source_info)
        return sources

    @staticmethod
    def _build_response(
        *,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: Optional[float],
        session: ConversationSession,
        document_id: Optional[int],
        max_chapter: Optional[int],
        include_reference: bool,
        llm_provider: Optional[str] = None,
        llm_calls: int = 0,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_used": len(sources),
            "session_id": session.session_id,
            "conversation_length": len(session.messages),
            "context_used": len(session.messages) > 2,
            "filtered_to_document": document_id,
            "spoiler_filter_active": max_chapter is not None,
            "max_chapter": max_chapter,
            "include_reference": include_reference,
            "llm_provider": llm_provider,
            "llm_calls": llm_calls,
        }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self.memory_manager.sessions:
            return []

        session = self.memory_manager.sessions[session_id]
        return [
            {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat(),
            }
            for msg in session.messages
        ]

    def clear_conversation(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self.memory_manager.sessions:
            del self.memory_manager.sessions[session_id]
        return True

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active conversation sessions."""
        return [s.get_summary() for s in self.memory_manager.sessions.values()]


# Global instance
context_aware_rag = None


def initialize_context_aware_rag(base_rag_service):
    """Initialize the context-aware RAG system."""
    global context_aware_rag
    context_aware_rag = ContextAwareRAG(base_rag_service)
    logger.info("✅ Conversational RAG initialized")
    return context_aware_rag
