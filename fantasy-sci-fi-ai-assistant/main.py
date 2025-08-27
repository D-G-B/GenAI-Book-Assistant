"""
Entry point for FastAPI application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import chat_routes, documents_routes
from app.database import Base, engine
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    print("🔑 Checking API keys...")
    settings.validate_api_keys()

    yield
    print("Application shutdown.")

app = FastAPI(
    title="RAG Assistant",
    description="A RAG-powered document chat assistant",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(documents_routes.router, prefix="/api/v1")
app.include_router(chat_routes.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "RAG Assistant API",
        "docs": "/docs",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "endpoints": [
            "/api/v1/documents/list",
            "/api/v1/documents/upload",
            "/api/v1/documents/{id}/process",
            "/api/v1/chat/ask",
            "/api/v1/chat/status"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
