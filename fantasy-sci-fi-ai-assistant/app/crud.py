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
from app import models  # ORM models (to be created later in database.py or models.py)

# Example placeholder function
def get_all_documents(db: Session):
    """
    Retrieve all lore documents from the database.
    """
    return db.query(models.LoreDocument).all()

# More functions will go here as the database models are defined
