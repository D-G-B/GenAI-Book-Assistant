"""
FastAPI application with RAG and conversational memory
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import chat_routes, documents_routes, conversational_routes
from app.database import Base, engine, SessionLocal, LoreDocument
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup...")
    Base.metadata.create_all(bind=engine)
    settings.validate_api_keys()

    # Initialize enhanced service (this creates document_manager internally)
    from app.services.enhanced_rag_service import enhanced_rag_service

    print("✅ Enhanced RAG service ready")

    # Process any unprocessed documents
    db = SessionLocal()
    try:
        docs = db.query(LoreDocument).all()
        if docs:
            print(f"\n📚 Found {len(docs)} documents in database")

            # The document manager will automatically skip already-processed documents
            for doc in docs:
                # Check if already processed
                if enhanced_rag_service.document_manager.is_processed(doc.id):
                    print(f"   ✓ {doc.title} (already processed)")
                else:
                    print(f"   Processing: {doc.title}")
                    await enhanced_rag_service.document_manager.add_document(db, doc.id)

            print("✅ Document processing complete\n")
        else:
            print("📝 No documents found in database\n")
    finally:
        db.close()

    yield
    print("👋 Application shutdown")

app = FastAPI(
    title="RAG Document Assistant",
    description="RAG-powered chat assistant with conversational memory",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API routes
app.include_router(documents_routes.router, prefix="/api/v1")
app.include_router(chat_routes.router, prefix="/api/v1")
app.include_router(conversational_routes.router, prefix="/api/v1")

# Frontend
@app.get("/")
async def root(request: Request):
    """Serve the main frontend application"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    # so i dont forget : ->>>> source .venv/bin/activate  then ->>> uv run uvicorn main:app --reload
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)