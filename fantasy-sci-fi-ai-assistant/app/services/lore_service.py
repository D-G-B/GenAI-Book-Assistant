"""
Core RAG service for the lore companion.

This service handles:
1. Document chunking and processing
2. Creating embeddings for document chunks
3. Storing chunks in a simple vector database (FAISS)
4. Retrieving relevant chunks for questions
5. Generating answers using LLM with retrieved context
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sqlalchemy.orm import Session

from app import database, crud, lore_schemas
from app.config import settings


class SimpleLoreService:
    def __init__(self):
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        # Initialize FAISS index our 'google' for vector db
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_metadata = []

        # Simple in-memory storage
        self.documents = {}
        self.chunks = {}
        self.chunk_counter = 0

    def chunk_text(self, text, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks of size `chunk_size`
        Simple implementation - splits by sentence when possible
        """
        if not text or len(text.strip()) == 0:
            return []

        # Simple sentence-aware chunking - just for testing purposes
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Add sentence to current chunk
            test_chunk = current_chunk + sentence + ". "

            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk and start new one with overlap
                chunks.append(current_chunk.strip())

                # Create overlap by keeping last part of chunk
                words = current_chunk.split()
                overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
                current_chunk = " ".join(overlap_words) + " " + sentence + ". "
            else:
                current_chunk = test_chunk

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter very short chunks

    async def process_document(self, db: Session, document_id: int) -> bool:
        """
        Process a document by chunking, creating embeddings, and storing in vector database.

        Args:
            self: The SimpleLoreService instance
            db: SQLAlchemy database session for document retrieval
            document_id: ID of the document to process from the database

        Returns:
            bool: True if document was successfully processed, False if processing failed
        """
        try:
            # Get document from database
            doc = db.query(database.LoreDocument).filter(document_id == database.LoreDocument.id).first()
            if not doc:
                return False

            # Chunk document
            chunks = self.chunk_text(doc.content)
            if not chunks:  #
                return False

            print(f"Processing document '{doc.title}' into {len(chunks)} chunks...")

            # Create embeddings for all chunks
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

            print(f"✅ Successfully processed '{doc.title}' - {len(chunks)} chunks indexed ✅")
            return True

        except Exception as e:
            print(f"❌ Error processing document {document_id}: {str(e)} ❌")
            return False

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most relevant chunks for a given query using vector similarity search.

        Args:
            self: The SimpleLoreService instance
            query: User question or search query
            top_k: Maximum number of relevant chunks to return

        Returns:
            List[Dict[str, Any]]: List of chunk metadata with similarity scores, sorted by relevance
        """
        if self.index.ntotal == 0:
            return []

        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode(query)

            # Search FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))

            # Get chunk metadata for results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    chunk_info = self.chunk_metadata[idx].copy()
                    chunk_info['similarity_score'] = float(1 / (1 + distance))
                    results.append(chunk_info)

            return results

        except Exception as e:
            print(f"❌ Error searching chunks: {str(e)} ❌")
            return []

    async def ask_question(self, question: str, db: Session) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate answer using LLM.

        Args:
            self: The SimpleLoreService instance
            question: User's question about the lore documents
            db: SQLAlchemy database session for storing query results

        Returns:
            Dict[str, Any]: Dictionary containing answer, sources, confidence score, and metadata
        """
        if not question.strip():
            return {"error": "Question cannot be empty"}

        try:
            # Step 1: Retrieve relevant chunks
            relevant_chunks = self.search_chunks(question, top_k=3)

            if not relevant_chunks:
                return {
                    "answer": "I don't have any relevant information about that topic. Please upload some documents first!",
                    "sources": [],
                    "confidence": 0.0
                }

            # Step 2: Build context from chunks
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

            # Step 3: Create prompt for LLM
            system_prompt = """You are a knowledgeable lore companion for science fiction and fantasy worlds. 
    Your job is to answer questions about fictional universes using only the provided context.

    Rules:
    - Only use information from the provided context
    - If the context doesn't contain enough information, say so
    - Be specific and cite which source you're drawing from
    - Keep answers concise but informative
    - If asked about something not in the context, politely explain you don't have that information"""

            user_prompt = f"""Context from lore documents:
    {context}

    Question: {question}

    Please answer based only on the provided context. If the context doesn't contain relevant information, please say so."""

            # Step 4: Get answer from LLM (using OpenAI for MVP)
            if not settings.OPENAI_API_KEY:
                return {"error": "OpenAI API key not configured"}

            # Simple OpenAI call (we'll implement proper LLM service later)
            import openai
            client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            answer = response.choices[0].message.content

            # Step 5: Store query in database
            query_record = database.LoreQuery(
                question=question,
                answer=answer,
                sources=json.dumps(sources),
                model_used="gpt-3.5-turbo",
                cost=0.001  # Rough estimate for MVP
            )

            db.add(query_record)
            db.commit()

            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.8,  # Simple confidence score for MVP
                "chunks_used": len(relevant_chunks)
            }

        except Exception as e:
            print(f"❌ Error generating answer: {str(e)} ❌")
            return {"error": f"Failed to generate answer: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """
        Return status information about the lore service state.

        Args:
            self: The SimpleLoreService instance

        Returns:
            Dict[str, Any]: Dictionary with service status including document count, chunk count, and model info
        """
        return {
            "documents_loaded": len(self.documents),
            "total_chunks": self.index.ntotal,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "FAISS",
            "status": "ready" if self.index.ntotal > 0 else "no_documents"
        }


# Global instance
lore_service = SimpleLoreService()
