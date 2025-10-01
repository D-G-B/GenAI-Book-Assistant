"""
LangChain-enhanced RAG service that lives alongside your original.
This demonstrates the key improvements without overwhelming complexity.
"""

from typing import List, Dict, Any
from sqlalchemy.orm import Session

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.database import LoreDocument
from app.config import settings


class LangChainRAGService:
    """
    Enhanced RAG using LangChain - compare this to your SimpleRAGService!
    """

    def __init__(self):
        print("🚀 Initializing LangChain RAG Service...")

        # 1. Embeddings (similar to your embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        # 2. Text Splitter (better than your chunk_text method)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical!
        )

        # 3. Vector Store (replaces your FAISS index + metadata tracking)
        self.vector_store = None

        # 4. LLM (replaces your manual OpenAI/Gemini calls)
        self.llm = self._initialize_llm()

        print("✅ LangChain RAG Service initialized")

    def _initialize_llm(self):
        """Initialize LLM - much cleaner than manual API calls"""
        if settings.GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(
                model=settings.DEFAULT_GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3
            )
        elif settings.OPENAI_API_KEY:
            return ChatOpenAI(
                model_name=settings.DEFAULT_OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3
            )
        return None

    async def process_document(self, db: Session, document_id: int) -> bool:
        """
        Compare this to your process_document method!

        YOUR METHOD (~50 lines):
        - Manual chunking logic
        - Manual embedding creation
        - Manual FAISS index management
        - Manual metadata tracking

        LANGCHAIN METHOD (~20 lines):
        - Automatic chunking
        - Automatic embedding
        - Automatic indexing
        - Automatic metadata
        """
        try:
            # Get document (same as yours)
            doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if not doc or not doc.content:
                return False

            print(f"📄 Processing: {doc.title}")

            # LANGCHAIN MAGIC #1: Smart Text Splitting
            # Instead of your manual sentence splitting, this:
            # - Respects paragraph boundaries
            # - Handles edge cases
            # - Creates proper overlap
            chunks = self.text_splitter.split_text(doc.content)

            # LANGCHAIN MAGIC #2: Document Objects
            # Instead of separate chunk_metadata list and chunks dict, this:
            # - Bundles text + metadata together
            # - Type-safe
            # - Works with all LangChain tools
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'document_id': document_id,
                        'document_title': doc.title,
                        'chunk_index': i
                    }
                )
                for i, chunk in enumerate(chunks)
            ]

            # LANGCHAIN MAGIC #3: Vector Store Management
            # Instead of manual FAISS index + embedding + metadata tracking:
            if self.vector_store is None:
                # Create new store (embeds everything automatically!)
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add to existing store
                self.vector_store.add_documents(documents)

            print(f"✅ Processed {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    async def ask_question(self, question: str, db: Session) -> Dict[str, Any]:
        """
        Compare this to your ask_question method!

        YOUR METHOD (~80 lines):
        - Manual search_chunks call
        - Manual context building
        - Manual prompt construction
        - Manual LLM API calls (different for each provider)
        - Manual result formatting

        LANGCHAIN METHOD (~30 lines):
        - Create chain once
        - Call chain
        - Done!
        """
        if not self.vector_store or not self.llm:
            return {"error": "Service not ready"}

        try:
            # LANGCHAIN MAGIC #4: Prompt Templates
            # Instead of hardcoded f-strings, templates are:
            # - Reusable
            # - Testable
            # - Easy to modify
            prompt_template = """You are a knowledgeable assistant for fantasy/sci-fi documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Only use information from the provided context
- Be specific and cite sources
- If unsure, say so

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # LANGCHAIN MAGIC #5: Retrieval Chains
            # This ONE object handles:
            # - Retrieving relevant chunks (your search_chunks)
            # - Building context (your manual join)
            # - Calling LLM (your manual API calls)
            # - Formatting response
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Stuffs all docs into context
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            # LANGCHAIN MAGIC #6: Async Execution
            # One line does everything!
            result = qa_chain({"query": question})

            # Format response (similar to yours)
            sources = []
            for doc in result.get('source_documents', []):
                sources.append({
                    "document_title": doc.metadata.get('document_title'),
                    "chunk_index": doc.metadata.get('chunk_index'),
                    "similarity_score": 0.85  # Simplified
                })

            return {
                "answer": result['result'],
                "sources": sources,
                "confidence": 0.8,
                "chunks_used": len(sources)
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Simple status check"""
        chunk_count = 0
        if self.vector_store:
            try:
                chunk_count = self.vector_store.index.ntotal
            except:
                pass

        return {
            "service": "LangChain Enhanced",
            "total_chunks": chunk_count,
            "llm_available": self.llm is not None,
            "status": "ready" if chunk_count > 0 and self.llm else "not_ready"
        }


# Create a global instance (just like your rag_service)
langchain_rag_service = LangChainRAGService()