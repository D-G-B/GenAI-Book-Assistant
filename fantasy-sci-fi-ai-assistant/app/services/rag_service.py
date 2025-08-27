"""
Core RAG service for the chat assistant.
"""

import os
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sqlalchemy.orm import Session

from app.database import LoreDocument, LoreQuery
from app.config import settings

import openai
import google.generativeai as genai


class SimpleRAGService:
    def __init__(self):
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_metadata = []

        # Simple in-memory storage
        self.documents = {}
        self.chunks = {}
        self.chunk_counter = 0

    def chunk_text(self, text, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or len(text.strip()) == 0:
            return []

        # Simple sentence-aware chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + sentence + ". "

            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Create overlap
                words = current_chunk.split()
                overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
                current_chunk = " ".join(overlap_words) + " " + sentence + ". "
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]

    async def process_document(self, db: Session, document_id: int) -> bool:
        """Process a document by chunking and creating embeddings."""
        try:
            # Get document from database
            doc = db.query(LoreDocument).filter(LoreDocument.id == document_id).first()
            if not doc:
                return False

            # Chunk document
            chunks = self.chunk_text(doc.content)
            if not chunks:
                return False

            print(f"Processing document '{doc.title}' into {len(chunks)} chunks...")

            # Create embeddings
            embeddings = self.embedding_model.encode(chunks)

            # Add to FAISS index
            if embeddings.size > 0:
                self.index.add(embeddings.astype('float32'))

            # Store chunk metadata
            for i, chunk in enumerate(chunks):
                chunk_id = self.chunk_counter
                self.chunk_metadata.append({
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'document_title': doc.title,
                    'chunk_text': chunk,
                    'chunk_index': i
                })
                self.chunks[chunk_id] = chunk
                self.chunk_counter += 1

            # Store document reference
            self.documents[document_id] = {
                'title': doc.title,
                'filename': doc.filename,
                'chunk_count': len(chunks)
            }

            print(f"✅ Successfully processed '{doc.title}' - {len(chunks)} chunks indexed")
            return True

        except Exception as e:
            print(f"❌ Error processing document {document_id}: {str(e)}")
            return False

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find relevant chunks using vector similarity search."""
        if self.index.ntotal == 0:
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    chunk_info = self.chunk_metadata[idx].copy()
                    chunk_info['similarity_score'] = float(1 / (1 + distance))
                    results.append(chunk_info)

            return results

        except Exception as e:
            print(f"❌ Error searching chunks: {str(e)}")
            return []

    async def ask_question(self, question: str, db: Session) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate answer."""
        if not question.strip():
            return {"error": "Question cannot be empty"}

        try:
            # Retrieve relevant chunks
            relevant_chunks = self.search_chunks(question, top_k=3)

            if not relevant_chunks:
                return {
                    "answer": "I don't have any relevant information about that topic. Please upload some documents first!",
                    "sources": [],
                    "confidence": 0.0,
                    "chunks_used": 0
                }

            # Build context
            context_parts = []
            sources = []

            for chunk in relevant_chunks:
                context_parts.append(f"From '{chunk['document_title']}':\n{chunk['chunk_text']}")
                sources.append({
                    "document_title": chunk['document_title'],
                    "chunk_index": chunk['chunk_index'],
                    "similarity_score": chunk['similarity_score']
                })

            context = "\n\n---\n\n".join(context_parts)

            # Create prompts
            system_prompt = """You are a knowledgeable assistant for documents.
Answer questions using only the provided context.

Rules:
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite which source you're drawing from
- Keep answers concise but informative"""

            user_prompt = f"""Context from documents:
{context}

Question: {question}

Please answer based only on the provided context."""

            answer = ""
            model_used = ""

            # Check which model to use
            if settings.DEFAULT_GEMINI_MODEL and settings.GOOGLE_API_KEY:
                model_used = settings.DEFAULT_GEMINI_MODEL
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                model = genai.GenerativeModel(model_name=model_used)

                chat_session = model.start_chat()
                response = await chat_session.send_message_async(user_prompt)
                answer = response.text

            elif settings.DEFAULT_OPENAI_MODEL and settings.OPENAI_API_KEY:
                model_used = settings.DEFAULT_OPENAI_MODEL
                client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

                response = await client.chat.completions.create(
                    model=model_used,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            else:
                return {"error": "No valid API key or default model configured"}

            # Store query in database
            query_record = LoreQuery(
                question=question,
                answer=answer,
                sources=json.dumps(sources),
                model_used=model_used,
                cost=0.001
            )

            db.add(query_record)
            db.commit()

            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.8,
                "chunks_used": len(relevant_chunks)
            }

        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return {"error": f"Failed to generate answer: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """Return service status information."""
        return {
            "documents_loaded": len(self.documents),
            "total_chunks": self.index.ntotal,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "FAISS",
            "status": "ready" if self.index.ntotal > 0 else "no_documents"
        }

# Global instance
rag_service = SimpleRAGService()
