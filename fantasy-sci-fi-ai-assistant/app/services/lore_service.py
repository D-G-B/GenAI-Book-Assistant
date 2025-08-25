"""
Core RAG service for the lore companion.

This service handles:
1. Document chunking and processing
2. Creating embeddings for document chunks
3. Storing chunks in a simple vector database (FAISS)
4. Retrieving relevant chunks for questions
5. Generating answers using LLM with retrieved context
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class SimpleLoreService:
    def __init__(self):
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        # Initialize FAISS index our 'google' for vector db
        self. index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_metadata = []

        # Simple in-memory storage
        self.documents = {}
        self.chunks = {}
        self.chunk_counter = 0

    def chunk_text(self,text, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks of size `chunk_size`
        Simple implementation - splits by sentence when possible
        """
        if not text or len(text.strip()) == 0:
            return []

        # Simple sentence-aware chunking
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

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks
