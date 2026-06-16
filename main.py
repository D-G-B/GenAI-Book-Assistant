"""
FastAPI application with RAG and conversational memory
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.api import auth_routes, chat_routes, conversational_routes, documents_routes
from app.config import settings
from app.database import Base, LoreDocument, SessionLocal, engine

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application startup...")
    Base.metadata.create_all(bind=engine)
    settings.validate_api_keys()

    # Fail closed on a forgeable signing key: a known/empty secret lets anyone mint
    # a session cookie for any user. Allowed only in explicit local-dev modes.
    if settings.SESSION_SECRET_KEY in ("", "dev-insecure-change-me"):
        if settings.DEBUG or settings.DEV_AUTH_BYPASS:
            logger.warning(
                "SESSION_SECRET_KEY is unset/default — fine for local dev, but sessions are "
                "forgeable. Set a real value before deploying."
            )
        else:
            raise RuntimeError(
                "SESSION_SECRET_KEY is unset or the insecure default. Refusing to start. "
                'Set a strong value (python -c "import secrets; print(secrets.token_hex(32))"), '
                "or set DEBUG=true / DEV_AUTH_BYPASS=true for local dev."
            )

    if settings.DEV_AUTH_BYPASS:
        logger.warning(
            "DEV_AUTH_BYPASS is ON — every request authenticates as the dev user. "
            "NEVER enable this in a deployment."
        )

    # Initialize enhanced service (this creates document_manager internally)
    from app.services.enhanced_rag_service import enhanced_rag_service

    logger.info("✅ Enhanced RAG service ready")

    # Process any unprocessed documents
    db = SessionLocal()
    try:
        # Seed the fixed dev user (Increment 1 multi-tenancy stub) so the
        # LoreDocument.user_id FK is always satisfiable. Find-or-create, so it
        # stays harmless once Increment 2 swaps the stub for real login.
        from app.auth import get_or_create_dev_user
        get_or_create_dev_user(db)

        docs = db.query(LoreDocument).all()
        if docs:
            logger.info("📚 Found %d documents in database", len(docs))

            # The document manager will automatically skip already-processed documents
            for doc in docs:
                # Check if already processed
                if enhanced_rag_service.document_manager.is_processed(doc.id):
                    logger.info("   ✓ %s (already processed)", doc.title)
                else:
                    logger.info("   Processing: %s", doc.title)
                    if not await enhanced_rag_service.document_manager.add_document(db, doc.id):
                        logger.warning(
                            "   ⚠️ Failed to process '%s' (id=%d) at startup; see earlier logs for the reason",
                            doc.title, doc.id,
                        )

            logger.info("✅ Document processing complete")
        else:
            logger.info("📝 No documents found in database")
    finally:
        db.close()

    yield
    logger.info("👋 Application shutdown")


app = FastAPI(
    title="RAG Document Assistant",
    description="RAG-powered chat assistant with conversational memory",
    version="2.0.0",
    lifespan=lifespan,
)

# Signed-cookie session (Increment 2 auth). same_site="lax" lets the cookie ride
# the OAuth redirect back from Google; https_only adds the Secure flag and is
# driven by config (set SESSION_COOKIE_SECURE=true wherever the app is on HTTPS).
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET_KEY,
    same_site="lax",
    https_only=settings.SESSION_COOKIE_SECURE,
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Auth routes at top-level /auth (callback URL must match the Google redirect URI)
app.include_router(auth_routes.router)

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
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    # so i dont forget : ->>>> source .venv/bin/activate  then ->>> uv run uvicorn main:app --reload
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
