"""
Entry point for your FastAPI application.

Creates the FastAPI() app, registers routers from app/api/, and starts the server.

Usually contains:

    Imports of all routes (from app.api import lore, prompt_tester, documents)

    Root health check endpoint (/api/v1/health)

    Uvicorn startup code if run directly.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message" : "Hello World"}