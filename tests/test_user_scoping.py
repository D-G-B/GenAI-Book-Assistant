"""Tests for user-scoping of ingest stamping and document management.

- Ingest: _process_and_chunk stamps db_doc.user_id into every chunk's metadata
  (the seam the retrieval filter enforces against). Runs offline with the LLM
  chapter detector disabled so chunking is deterministic.
- list/delete: DocumentManager.list_all_documents / delete_document are scoped
  to the owning user. Uses an in-memory SQLite DB and a stub vector store
  manager (the manager is built via __new__ to skip the heavy __init__).
"""

import types

import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base, LoreDocument, User
from app.services.document_manager import DocumentManager


# ---------- ingest stamping ----------

def make_chunking_dm():
    dm = DocumentManager.__new__(DocumentManager)
    dm.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return dm


async def test_ingest_stamps_user_id_on_every_chunk(monkeypatch):
    """Every chunk produced from a document carries the document's owner."""
    monkeypatch.setattr(settings, "LLM_CHAPTER_DETECTION_ENABLED", False)
    dm = make_chunking_dm()

    content = (
        "The expedition set out before dawn, crossing the dunes while the air was "
        "still cold. By midday the wind had risen and the sand stung their faces.\n\n"
        "They made camp in the lee of a great rock and waited for the storm to pass, "
        "sharing what little water remained between them."
    )
    doc = types.SimpleNamespace(
        id=7, user_id=42, title="Owned Book", source_type="text", content=content
    )

    chunks = await dm._process_and_chunk(doc)

    assert chunks, "expected at least one chunk"
    assert all(c.metadata["user_id"] == 42 for c in chunks)


# ---------- list / delete scoping ----------

@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def make_manager_dm():
    """DocumentManager with the DB-facing methods live but the vector store stubbed."""
    dm = DocumentManager.__new__(DocumentManager)
    dm.vector_store_manager = types.SimpleNamespace(
        is_deleted=lambda _id: False,
        soft_delete_document=lambda _id: None,
    )
    dm.processed_documents = {}
    dm._save_manifest = lambda: None
    return dm


def seed(db):
    alice = User(username="alice", email="a@example.com")
    bob = User(username="bob", email="b@example.com")
    db.add_all([alice, bob])
    db.commit()
    db.add_all([
        LoreDocument(user_id=alice.id, title="A1", filename="a1.txt"),
        LoreDocument(user_id=alice.id, title="A2", filename="a2.txt"),
        LoreDocument(user_id=bob.id, title="B1", filename="b1.txt"),
    ])
    db.commit()
    return alice, bob


def test_list_is_user_scoped(db):
    alice, bob = seed(db)
    dm = make_manager_dm()

    assert {d["title"] for d in dm.list_all_documents(db, user_id=alice.id)} == {"A1", "A2"}
    assert {d["title"] for d in dm.list_all_documents(db, user_id=bob.id)} == {"B1"}


def test_delete_blocks_foreign_user(db):
    alice, bob = seed(db)
    dm = make_manager_dm()
    alice_doc = db.query(LoreDocument).filter_by(title="A1").first()

    # Bob cannot delete Alice's document.
    assert dm.delete_document(db, alice_doc.id, user_id=bob.id) is False
    # It is still present.
    assert db.query(LoreDocument).filter_by(id=alice_doc.id).first() is not None


def test_delete_allows_owner(db):
    alice, _bob = seed(db)
    dm = make_manager_dm()
    alice_doc = db.query(LoreDocument).filter_by(title="A1").first()

    assert dm.delete_document(db, alice_doc.id, user_id=alice.id) is True
    assert db.query(LoreDocument).filter_by(id=alice_doc.id).first() is None
