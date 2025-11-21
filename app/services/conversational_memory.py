"""
Conversational memory system for context-aware conversations

"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


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
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        }
        self.messages.append(message)
        self.last_activity = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this conversation."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'message_count': len(self.messages)
        }


class ConversationMemoryManager:
    """Manages multiple conversation sessions."""

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions

    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            # Clean up old sessions if at limit
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_session()

            self.sessions[session_id] = ConversationSession(session_id, user_id)

        return self.sessions[session_id]

    def _cleanup_oldest_session(self):
        """Remove the oldest session."""
        if not self.sessions:
            return

        oldest_session_id = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].last_activity
        )
        del self.sessions[oldest_session_id]

    def create_memory(self, k: int = 5) -> ConversationBufferWindowMemory:
        """Create a LangChain memory instance that keeps last k exchanges."""
        return ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer"
        )

    def create_conversational_chain(self, retriever, llm):
        """Create a conversational retrieval chain with memory."""

        memory = self.create_memory(k=5)

        # Custom prompt for conversational context
        condense_question_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that incorporates relevant context from the conversation history.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone Question:"""

        condense_question_prompt = PromptTemplate.from_template(condense_question_template)

        # Create conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=True
        )

        return chain


class ContextAwareRAG:
    """RAG system with conversational memory."""

    def __init__(self, base_rag_service):
        self.base_rag = base_rag_service
        self.memory_manager = ConversationMemoryManager()
        self.active_chains: Dict[str, Any] = {}  # Session ID -> Chain

    async def ask_with_context(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ask a question with conversational context, optionally filtered to a document."""

        if not question.strip():
            return {"error": "Question cannot be empty"}

        if not self.base_rag.llm:
            return {"error": "No language model configured"}

        if not self.base_rag.vector_store_manager.vector_store:
            return {"error": "No documents available"}

        try:
            # Get or create session
            session = self.memory_manager.get_or_create_session(session_id, user_id)

            # Create chain key that includes document_id to maintain separate chains per filter
            chain_key = f"{session_id}_{document_id}" if document_id else session_id

            # Create or get conversational chain for this session/filter combination
            if chain_key not in self.active_chains:
                # Use vector store manager's get_retriever method (handles filtering)
                retriever = self.base_rag.vector_store_manager.get_retriever(
                    k=4,
                    document_id=document_id
                )

                if retriever is None:
                    return {"error": "Vector store not available"}

                if document_id is not None:
                    print(f"💬 Conversational search in document ID: {document_id}")

                chain = self.memory_manager.create_conversational_chain(
                    retriever=retriever,
                    llm=self.base_rag.llm
                )
                self.active_chains[chain_key] = chain
            else:
                chain = self.active_chains[chain_key]

            # Add question to session history
            session.add_message('human', question)

            print(f"💬 Conversational question (Session: {session_id[:8]}...): {question}")

            # Run the conversational chain
            result = chain({"question": question})

            # Extract information
            answer = result.get('answer', 'No answer generated')
            source_docs = result.get('source_documents', [])

            # Process sources
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "document_title": doc.metadata.get('document_title', 'Unknown'),
                    "chunk_index": doc.metadata.get('chunk_index', i),
                    "similarity_score": 0.85
                })

            # Add answer to session history
            session.add_message('assistant', answer)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.8,
                "chunks_used": len(sources),
                "session_id": session_id,
                "conversation_length": len(session.messages),
                "context_used": len(session.messages) > 2,  # True if more than just this Q&A
                "filtered_to_document": document_id
            }

        except Exception as e:
            print(f"❌ Error in conversational question: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self.memory_manager.sessions:
            return []

        session = self.memory_manager.sessions[session_id]
        return [
            {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat()
            }
            for msg in session.messages
        ]

    def clear_conversation(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self.memory_manager.sessions:
            del self.memory_manager.sessions[session_id]

        # Clear all chains associated with this session
        keys_to_delete = [k for k in self.active_chains.keys() if k.startswith(session_id)]
        for key in keys_to_delete:
            del self.active_chains[key]

        return True

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active conversation sessions."""
        return [
            session.get_summary()
            for session in self.memory_manager.sessions.values()
        ]


# Global instance - will be initialized with base RAG service
context_aware_rag = None

def initialize_context_aware_rag(base_rag_service):
    """Initialize the context-aware RAG system."""
    global context_aware_rag
    context_aware_rag = ContextAwareRAG(base_rag_service)
    print("✅ Conversational RAG initialized")
    return context_aware_rag