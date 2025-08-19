"""
crud.py
-------
CRUD = Create, Read, Update, Delete operations.

This file contains reusable database functions for our models.
Separating them here keeps the API route files clean — they just call these functions.

Example functions we will add:
- create_lore_document(db, document_data)
- get_all_documents(db)
- delete_document(db, document_id)
- create_prompt_test_result(db, result_data)
- fetch_prompt_tests(db, filters)
"""

from sqlalchemy.orm import Session
from app import database, lore_schemas # ORM models (to be created later in database.py or prompt_schemas.py)

def get_all_documents(db: Session):
    """
    Retrieve all lore documents from the database.
    """
    return db.query(database.LoreDocument).all()

# More functions will go here as the database models are defined

def  create_lore_document(db: Session, document: lore_schemas.LoreDocumentCreate):
    """
    Creates a new lore document in the database.

    Args:
        db: The SQLAlchemy database session
        documentdata: A Pydantic schema object with the document's data.
    Returns:
        The newly created LoreDocument ORM object
    """
    new_lore_doc = database.LoreDocument(**document.model_dump())

    try:
        db.add(new_lore_doc)
        db.commit()
        db.refresh(new_lore_doc)
        return(new_lore_doc)
    except Exception:
        db.rollback()
        return None

def delete_document(db: Session, document_id: int):
    """
    Deletes a lore document from the database.

    Args:
        db (Session): The SQLAlchemy database session
        document_id (int): the document id of the document to be deleted
    """

    doc_to_del = db.query(database.LoreDocument).filter(database.LoreDocument.id == document_id).first()

    if not doc_to_del:
        return None

    try:
        db.delete(doc_to_del)
        db.commit()
        return doc_to_del
    except Exception:
        db.rollback()
        return None
