"""
Entry point for your FastAPI application.

Creates the FastAPI() app, registers routers from app/api/, and starts the server.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import documents, lore
from app.database import Base, engine
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on application startup
    print("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    # Validate API keys
    print("🔑 Checking API keys...")
    settings.validate_api_keys()

    yield # The application will start here
    # This prints on application shutdown
    print("Application shutdown.")

# We pass the lifespan context manager to the FastAPI app
app = FastAPI(
    title="GenAI Sci-Fi Assistant",
    description="A RAG-powered lore companion and multi-model prompt tester",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(documents.router, prefix="/api/v1")
app.include_router(lore.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "GenAI Sci-Fi Assistant API",
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
            "/api/v1/lore/ask",
            "/api/v1/lore/status"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)