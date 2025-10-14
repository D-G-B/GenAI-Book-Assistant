"""
Entry point - now includes BOTH original and enhanced services
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import chat_routes, documents_routes
from app.api import comparison_routes, conversational_routes
from app.database import Base, engine, SessionLocal, LoreDocument
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup...")
    Base.metadata.create_all(bind=engine)
    settings.validate_api_keys()

    # Initialize BOTH services
    from app.services.rag_service import rag_service
    from app.services.langchain_rag_service import langchain_rag_service

    print("✅ Original RAG service ready")
    print("✅ LangChain RAG service ready")

    # Process any existing documents on startup
    db = SessionLocal()  # This creates a database session
    try:
        docs = db.query(LoreDocument).all()
        if docs:
            print(f"\n📚 Found {len(docs)} existing documents")
            for doc in docs:
                print(f"   Processing: {doc.title}")
                # Process with both services
                await rag_service.process_document(db, doc.id)
                await langchain_rag_service.process_document(db, doc.id)
            print("✅ All documents processed by both services\n")
    finally:
        db.close()  # Always close the session when done

    yield
    print("👋 Application shutdown")

app = FastAPI(
    title="RAG Assistant - Original + LangChain",
    description="Compare your original RAG with LangChain enhanced version",
    version="0.2.0",
    lifespan=lifespan
)

# Original routes
app.include_router(documents_routes.router, prefix="/api/v1")
app.include_router(chat_routes.router, prefix="/api/v1")

# New Comparison routes
app.include_router(comparison_routes.router, prefix="/api/v1")

# Conversational Routes
app.include_router(conversational_routes.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "RAG Assistant with Original + LangChain",
        "original_endpoints": [
            "/api/v1/chat/ask",
            "/api/v1/chat/status"
        ],
        "comparison_endpoints": [
            "/api/v1/compare/ask-original",
            "/api/v1/compare/ask-langchain",
            "/api/v1/compare/ask-both"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.2.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
