"""
Core RAG service for the lore companion.

This service handles:
1. Document chunking and processing
2. Creating embeddings for document chunks
3. Storing chunks in a simple vector database (FAISS)
4. Retrieving relevant chunks for questions
5. Generating answers using LLM with retrieved context
"""

