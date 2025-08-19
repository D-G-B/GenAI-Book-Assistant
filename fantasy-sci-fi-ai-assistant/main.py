"""
Entry point for your FastAPI application.

Creates the FastAPI() app, registers routers from app/api/, and starts the server.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import documents
from app.database import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on application startup
    print("Application startup: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
    yield # The application will start here
    # This code runs on application shutdown
    print("Application shutdown.")

# We pass the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)
app.include_router(documents.router)

@app.get("/")
async def root():
    return {"message" : "Hello World"}
