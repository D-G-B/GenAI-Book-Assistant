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

        #Initialize conversational features
        self._setup_conversational_rag()

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

    def _setup_conversational_rag(self):
        """Initialize conversational RAG system."""
        from app.services.conversational_memory import initialize_context_aware_rag

        if self.llm:
            self.context_aware_rag = initialize_context_aware_rag(self)
        else:
            self.context_aware_rag = None
            print("⚠️ Conversational features not available (no LLM)")

    
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



        """Enhanced question answering with LangChain"""
        if not question.strip():
            return {"error": "Question cannot be empty"}

        if not self.vector_store:
            return {
                "answer": "I don't have any documents to search through. Please upload and process some documents first!",
                "sources": [],
                "confidence": 0.0,
                "chunks_used": 0
            }

        if not self.llm:
            return {"error": "No language model configured"}

        try:
            # Create prompt template
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

            # Create retrieval chain
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            # Execute the chain
            print(f"🤔 Processing question: {question}")
            result = qa_chain({"query": question})

            # Extract answer safely
            answer = result.get('result', 'No answer generated')
            source_docs = result.get('source_documents', [])

            # Format sources
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "document_title": doc.metadata.get('document_title', 'Unknown'),
                    "chunk_index": doc.metadata.get('chunk_index', i),
                    "similarity_score": 0.85
                })

            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.8,
                "chunks_used": len(sources)
            }

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get service status information"""
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
            "vector_store_ready": self.vector_store is not None,
            "status": "ready" if chunk_count > 0 and self.llm else "not_ready"
    }

# Create a global instance (just like your rag_service)
langchain_rag_service = LangChainRAGService()
