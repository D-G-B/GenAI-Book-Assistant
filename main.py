"""
Production FastAPI application with enhanced RAG and conversational memory
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

    # Initialize enhanced service
    from app.services.enhanced_rag_service import enhanced_rag_service

    print("✅ Enhanced RAG service ready")

    # Process existing documents
    db = SessionLocal()
    try:
        docs = db.query(LoreDocument).all()
        if docs:
            print(f"\n📚 Found {len(docs)} existing documents")
            for doc in docs:
                print(f"   Processing: {doc.title}")
                await enhanced_rag_service.process_document(db, doc.id)
            print("✅ All documents processed\n")
    finally:
        db.close()

    yield
    print("👋 Application shutdown")

app = FastAPI(
    title="Fantasy RAG Assistant",
    description="RAG-powered chat assistant with conversational memory",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API routesgem
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
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # reminder cause i always forget :::::: source .venv/bin/activate :::::::
    import uvicorn
    uvicorn.run(app, host="0.0git.0.0", port=8000, reload=True)