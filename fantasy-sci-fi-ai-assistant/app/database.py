"""
database.py
-----------
Handles the database connection setup and ORM model definitions.
This file is where we configure SQLAlchemy (or another ORM) and connect to SQLite for MVP.

Later, we can switch DATABASE_URL in `.env` to use PostgreSQL in production.

Main responsibilities:
- Create an SQLAlchemy engine and session maker
- Define Base class for ORM models
- Provide a dependency function `get_db()` for FastAPI endpoints
"""
from sqlalchemy import create_engine, Column, Integer,Numeric, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import os

# Load DB URL from environment (SQLite for now)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# For SQLite, `check_same_thread=False` is required for FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# SessionLocal will be used to create DB sessions in endpoints
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()

# Dependency for getting DB session in API routes
def get_db():
    """
    Yields a database session for use in API endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, unique=True)
    username = Column(String(50), nullable=False, index=True, unique=True)
    email = Column(String(120), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

class LoreDocument(Base):
    __tablename__ = "lore_documents"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    filename = Column(Text, nullable=False)
    content = Column(Text)
    source_type = Column(String(50))
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("lore_documents.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer) # order of chunk in document
    metadata = Column(JSON)

    document = relationship("LoreDocument", back_populates="chunks")

class LoreQuery(Base):
    __tablename__ = "lore_queries"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer =  Column(Text)
    sources = Column(JSON) # list of chunk ids or text
    model_used = Column(String(50))
    cost = Column(Numeric)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
