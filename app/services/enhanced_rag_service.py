"""
Enhanced RAG Service - Focuses on RAG queries and answer generation.
Supports simplified spoiler filtering with optional reference material.
"""

from typing import Dict, Any, Optional

from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.config import settings
from app.services.document_manager import DocumentManager
from app.services.vector_store_manager import VectorStoreManager


class EnhancedRAGService:
    """
    Enhanced RAG service focused on query processing and answer generation.
    """

    def __init__(self):
        print("🚀 Initializing Enhanced RAG Service...")

        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager()

        # Initialize document manager
        self.document_manager = DocumentManager(self.vector_store_manager)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Conversational features
        self.context_aware_rag = None
        self._setup_conversational_rag()

        print("✅ Enhanced RAG Service initialized")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on available API keys."""

        if settings.OPENAI_API_KEY and settings.DEFAULT_OPENAI_MODEL:
            print(f"✅ Using OpenAI: {settings.DEFAULT_OPENAI_MODEL}")
            return ChatOpenAI(
                model_name=settings.DEFAULT_OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=1000
            )
        elif settings.ANTHROPIC_API_KEY and settings.DEFAULT_CLAUDE_MODEL:
            print(f"✅ Using Claude: {settings.DEFAULT_CLAUDE_MODEL}")
            return ChatAnthropic(
                model=settings.DEFAULT_CLAUDE_MODEL,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.3,
                max_tokens=1000
            )
        elif settings.GOOGLE_API_KEY and settings.DEFAULT_GEMINI_MODEL:
            print(f"✅ Using Google Gemini: {settings.DEFAULT_GEMINI_MODEL}")
            return ChatGoogleGenerativeAI(
                model=settings.DEFAULT_GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3,
                max_tokens=1000
            )
        else:
            print("⚠️ No LLM configured - check your API keys")
            return None

    def _setup_conversational_rag(self):
        """Initialize conversational RAG system."""
        if self.llm:
            from app.services.conversational_memory import initialize_context_aware_rag
            self.context_aware_rag = initialize_context_aware_rag(self)
        else:
            print("⚠️ Conversational features not available (no LLM)")

    async def ask_question(
        self,
        question: str,
        document_id: Optional[int] = None,
        max_chapter: Optional[int] = None,
        include_reference: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            document_id: Optional document ID to filter search
            max_chapter: Optional max chapter for spoiler protection (None = no filter)
            include_reference: Include reference material when spoiler filter is active

        Returns:
            Dictionary with answer, sources, and metadata
        """

        if not question.strip():
            return {"error": "Question cannot be empty"}

        if not self.vector_store_manager.vector_store:
            return {
                "answer": "I don't have any documents to search through. Please upload and process some documents first!",
                "sources": [],
                "confidence": 0.0,
                "chunks_used": 0
            }

        if not self.llm:
            return {"error": "No language model configured. Please check your API keys."}

        try:
            # Create prompt template
            # [FIX] Refined prompt to allow logical synthesis without being Dune-specific
            prompt_template = """You are an expert Reading Companion and Lorekeeper.
Your goal is to help the user understand the world, remember characters, and track plotlines.

Context from the book/documents:
{context}

User's Question: {question}

Instructions:
1. **Role**: Act as a helpful guide. If asked "Who is X?", provide their identity, allegiance, and key relationships based on the context.
2. **Terminology**: If unique or technical terms appear in the context, define them briefly if relevant to the answer.
3. **Synthesis Allowed**: Base your answer *only* on the provided context, but **you may synthesize details** from multiple sections to form a complete answer (e.g., connecting family relationships mentioned in different places). Do not use outside knowledge.
4. **Spoilers**: Answer the specific question asked. Do not reveal major future plot twists unless explicitly asked.
5. **Clarity**: Be precise with spelling and relationships.

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Log what we're searching
            filter_info = []
            if document_id is not None:
                filter_info.append(f"document {document_id}")
            if max_chapter is not None:
                filter_info.append(f"chapters 1-{max_chapter}")
                if include_reference:
                    filter_info.append("+ reference material")

            if filter_info:
                print(f"🔍 Searching with filters: {', '.join(filter_info)}")
            else:
                print(f"🔍 Searching across all documents (no spoiler filter)")

            # Get retriever with filters
            # [FIX] We call get_retriever here with k=20 to ensure we have enough context
            # for the synthesis instruction to work effectively.
            retriever = self.vector_store_manager.get_retriever(
                k=20,
                document_id=document_id,
                max_chapter=max_chapter,
                include_reference=include_reference
            )

            if retriever is None:
                return {"error": "Vector store not available"}

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT, "verbose": True},
                return_source_documents=True,
                verbose=True
            )

            # Execute query
            print(f"🤔 Processing question: {question}")
            result = qa_chain({"query": question})

            # Extract information
            answer = result.get('result', 'No answer generated')
            source_docs = result.get('source_documents', [])

            # Format sources
            sources = []
            for i, doc in enumerate(source_docs):
                source_info = {
                    "document_title": doc.metadata.get('document_title', 'Unknown'),
                    "chunk_index": doc.metadata.get('chunk_index', i),
                    "similarity_score": 0.85
                }

                # Add chapter info if available
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

            # Calculate confidence
            confidence = min(0.95, 0.85 * 1.1)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(sources),
                "spoiler_filter_active": max_chapter is not None,
                "max_chapter": max_chapter,
                "include_reference": include_reference
            }

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

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