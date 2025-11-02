"""
Production-ready RAG service using LangChain with document loaders and conversational memory.
"""

from typing import List, Dict, Any, Optional

from langchain_community.chat_models import ChatAnthropic
from sqlalchemy.orm import Session
import time
import json
from pathlib import Path
from datetime import datetime

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

# document processor
from app.services.advanced_document_loaders import document_processor as advanced_doc_processor


class EnhancedRAGService:
    """RAG service with LangChain, document loaders, and conversational memory."""

    def __init__(self):
        print("🚀 Initializing Enhanced RAG Service...")

        # Vector store persistence path
        self.vector_store_path = Path("./faiss_index")
        self.manifest_path = Path("./faiss_index/manifest.json")

        # Core components
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        # Text splitter with intelligent? chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # Use the advanced document processor
        self.document_processor = advanced_doc_processor

        # Storage
        self.vector_store = None
        self.documents: List[Document] = []

        # Tracking - MUST initialize BEFORE loading vector store
        self.processed_documents = {}

        # Try to load existing vector store (will populate processed_documents)
        self._load_vector_store()

        # LLM
        self.llm = self._initialize_llm()

        # Conversational features
        self.context_aware_rag = None
        self._setup_conversational_rag()

        print("✅ RAG Service initialized with advanced document loaders")

    def _load_manifest(self) -> set:
        """Load the set of processed document IDs from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                    processed_ids = set(data.get('processed_document_ids', []))
                    print(f"📋 Loaded manifest: {len(processed_ids)} documents already processed")
                    return processed_ids
            except Exception as e:
                print(f"⚠️ Could not load manifest: {e}")
        return set()

    def _save_manifest(self):
        """Save the set of processed document IDs to disk."""
        try:
            # Ensure directory exists
            self.vector_store_path.mkdir(exist_ok=True)

            manifest_data = {
                'processed_document_ids': list(self.processed_documents.keys()),
                'last_updated': datetime.now().isoformat()
            }

            with open(self.manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)

            print(f"📋 Manifest saved: {len(self.processed_documents)} documents")
        except Exception as e:
            print(f"⚠️ Failed to save manifest: {e}")

    def _load_vector_store(self):
        """Load vector store from disk if it exists."""
        if self.vector_store_path.exists():
            try:
                print(f"📂 Loading existing vector store from {self.vector_store_path}")
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Load the manifest to know which documents are already processed
                processed_ids = self._load_manifest()
                for doc_id in processed_ids:
                    self.processed_documents[doc_id] = {
                        'loaded_from_disk': True
                    }

                # Get chunk count
                try:
                    chunk_count = self.vector_store.index.ntotal
                    print(f"✅ Loaded vector store with {chunk_count} existing chunks")
                except:
                    print(f"✅ Loaded vector store")
            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
                print("   Will create new vector store")
                self.vector_store = None

    def _save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(str(self.vector_store_path))
                self._save_manifest()
                print(f"💾 Vector store and manifest saved to {self.vector_store_path}")
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")

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
        elif settings.ANTHROPIC_API_KEY and settings.DEFAULT_ANTHROPIC_MODEL:
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

    async def process_document(self, db: Session, document_id: int) -> bool:
        """Process a document with enhanced capabilities using advanced loaders."""

        # Skip if already processed
        if document_id in self.processed_documents:
            print(f"⏭️  Document {document_id} already processed, skipping")
            return True

        start_time = time.time()

        try:
            # Get document from database
            db_doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if not db_doc or not db_doc.content:
                print(f"❌ Document {document_id} not found or has no content")
                return False

            print(f"📄 Processing: {db_doc.title} ({db_doc.filename})")

            # Check if content is a placeholder for failed extraction
            if db_doc.content.startswith('[PDF') and 'extraction failed' in db_doc.content:
                print(f"⚠️ Skipping document with failed extraction: {db_doc.filename}")
                return False

            # Create base metadata
            base_metadata = {
                'document_id': document_id,
                'document_title': db_doc.title,
                'source_type': db_doc.source_type or 'text'
            }

            # Use advanced document processor to handle different file types
            try:
                initial_docs = self.document_processor.process_content(
                    db_doc.content,
                    db_doc.filename,
                    base_metadata
                )
            except Exception as e:
                print(f"❌ Error with document processor: {e}")
                # Fallback to treating as plain text
                if len(db_doc.content.strip()) > 20:
                    initial_docs = [Document(page_content=db_doc.content, metadata=base_metadata)]
                else:
                    print(f"❌ Content too short or invalid")
                    return False

            if not initial_docs:
                print(f"❌ No valid content from document {document_id}")
                return False

            # Split into chunks and validate
            final_documents = []
            for document in initial_docs:
                # Skip empty documents
                if not document.page_content or not document.page_content.strip():
                    print(f"⚠️ Skipping empty document chunk")
                    continue

                # Skip very short content
                if len(document.page_content.strip()) < 20:
                    print(f"⚠️ Skipping very short content")
                    continue

                try:
                    chunks = self.text_splitter.split_text(document.page_content)
                except Exception as e:
                    print(f"⚠️ Error splitting text: {e}")
                    continue

                for i, chunk in enumerate(chunks):
                    # Validate chunk has actual content
                    if not chunk or not chunk.strip():
                        continue

                    # Skip chunks that are too short (less than 10 chars)
                    if len(chunk.strip()) < 10:
                        continue

                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })

                    final_documents.append(Document(
                        page_content=chunk.strip(),
                        metadata=chunk_metadata
                    ))

            if not final_documents:
                print(f"❌ No valid chunks created from document {document_id}")
                return False

            print(f"✅ Created {len(final_documents)} valid chunks")

            # Store documents
            self.documents.extend(final_documents)

            # Update or create vector store with better error handling
            try:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(final_documents, self.embeddings)
                    print(f"✅ Created vector store with {len(final_documents)} chunks")
                else:
                    # Add documents one at a time to catch individual failures
                    success_count = 0
                    for doc in final_documents:
                        try:
                            self.vector_store.add_documents([doc])
                            success_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to add chunk {doc.metadata.get('chunk_index', '?')}: {e}")
                            continue

                    if success_count > 0:
                        print(f"✅ Added {success_count}/{len(final_documents)} chunks to vector store")
                    else:
                        print(f"❌ Failed to add any chunks")
                        return False

            except Exception as e:
                print(f"❌ Error with vector store: {e}")
                import traceback
                traceback.print_exc()
                return False

            # Track processing - MUST do this BEFORE saving manifest
            processing_time = time.time() - start_time
            self.processed_documents[document_id] = {
                'title': db_doc.title,
                'filename': db_doc.filename,
                'chunk_count': len(final_documents),
                'processing_time': processing_time
            }

            # Save vector store to disk after processing (includes manifest)
            self._save_vector_store()

            print(f"✅ Processed '{db_doc.title}' in {processing_time:.2f}s - {len(final_documents)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error processing document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def ask_question(
        self,
        question: str,
        db: Session,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Answer a question using RAG, optionally filtered to a specific document."""
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
            return {"error": "No language model configured. Please check your API keys."}

        try:
            # Create prompt template
            prompt_template = """You are a knowledgeable assistant for fantasy and sci-fi documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Only use information from the provided context
- Be specific and cite sources when possible
- If the context doesn't contain enough information, say so clearly
- Keep answers informative but concise

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create retriever with optional document filter
            search_kwargs = {"k": 4}

            if document_id is not None:
                # Filter to specific document
                search_kwargs["filter"] = lambda metadata: metadata.get("document_id") == document_id
                print(f"🔍 Searching only in document ID: {document_id}")
            else:
                print(f"🔍 Searching across all documents")

            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            # Execute
            print(f"🤔 Processing question: {question}")
            result = qa_chain({"query": question})

            # Extract information
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

            # Calculate confidence
            avg_similarity = 0.85  # Simplified
            confidence = min(0.95, avg_similarity * 1.1)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(sources)
            }

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        chunk_count = 0
        if self.vector_store:
            try:
                chunk_count = self.vector_store.index.ntotal
            except:
                pass

        return {
            "documents_loaded": len(self.processed_documents),
            "total_chunks": chunk_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "FAISS",
            "llm_available": self.llm is not None,
            "conversational_available": self.context_aware_rag is not None,
            "status": "ready" if chunk_count > 0 and self.llm else "not_ready",
            "persistent_storage": self.vector_store_path.exists()
        }


# Global service instance
enhanced_rag_service = EnhancedRAGService()