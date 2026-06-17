"""Synthetic, copyright-free corpus for end-to-end retrieval tests.

Modeled on the confirmed Dune "Baron Harkonnen" failure so the tests exercise the
*real* shapes the production filter + retrieval see:
  - book divisions over continuous chapter numbering (1..8 across BOOK I/II),
  - an antagonist (Baron Vethran) alive and scheming across MANY chapters,
  - his death in a single late narrative chapter, written obliquely (no "died"),
  - a reference/appendix entry that states the death plainly (the spoiler-excluded
    copy of the fact),
  - chunks owned by a second user (tenancy) and a soft-deletable document.

`build_documents()` returns `langchain.schema.Document`s stamped with the exact
metadata keys production stamps (see DocumentManager._create_chunks), so the real
VectorStoreManager filter + source formatting behave identically.
"""

from typing import List

from langchain.schema import Document

DOC_TITLE = "The Reclamation of Kaaran"

# Owners
USER_A = 1  # the test's primary user (owns docs 1 and 3)
USER_B = 2  # a second tenant (owns doc 2)

# Document ids
DOC_MAIN = 1       # the "book"
DOC_OTHER_USER = 2  # owned by USER_B (tenancy)
DOC_DELETABLE = 3   # owned by USER_A, soft-deleted in one test

# Markers used by tests to identify specific chunks
DEATH_CHAPTER = 8          # late narrative chapter holding the (oblique) death scene
DEATH_MARKER = "Whisper's edge"   # appears ONLY in the death chunk
REFERENCE_MARKER = "Iron Baron"   # appears ONLY in the plain-statement reference chunk
SALT_GATE_MARKER = "Salt Gate"    # appears ONLY in the deletable doc's chunk
OTHER_USER_MARKER = "border banners"  # appears ONLY in USER_B's chunk

# Queries
QUERY_VETHRAN_PLAN = "What is Baron Vethran planning against House Ardent?"
QUERY_VETHRAN_FATE = "What happens to Baron Vethran in the end? Does he die?"
QUERY_THRONE = "the confrontation at the dais in the throne room of Kaaran"
QUERY_NOBLE_HOUSES = "record of House Vethran in the Almanak of the Noble Houses"
QUERY_SALT_GATE = "what is the Provisional Accord at the Salt Gate"


# (document_id, user_id, chapter_number, chapter_title, is_reference, page_content)
_RECORDS = [
    # ---- DOC_MAIN, BOOK I — THE GATHERING (chapters 1-4) ----
    (DOC_MAIN, USER_A, 1, "BOOK I — Chapter 1", False,
     "The desert world of Kaaran turned beneath two moons, its dunes rich with the "
     "pale spice called vael. House Ardent had governed the spice harvest for nine "
     "generations from the citadel at Far Reach."),
    (DOC_MAIN, USER_A, 2, "BOOK I — Chapter 2", False,
     "Baron Vethran smiled coldly in his keep on Giedan and plotted the downfall of "
     "House Ardent. He coveted the vael harvest of Kaaran and meant to take it by "
     "treachery, whatever the cost in blood."),
    (DOC_MAIN, USER_A, 3, "BOOK I — Chapter 3", False,
     "Vethran dispatched his agents and spies across Kaaran, buying informers in Far "
     "Reach and seeding rumor among the harvest crews. The Baron schemed late into the "
     "night, hungry for the moment House Ardent would fall."),
    (DOC_MAIN, USER_A, 4, "BOOK I — Chapter 4", False,
     "Young Lyra of House Ardent trained in the Whisper arts under the Reverend Keeper, "
     "learning the breath-control and the still mind that let an adept read a room of "
     "enemies and bend a single will."),
    # ---- DOC_MAIN, BOOK II — THE RECLAMATION (chapters 5-8) ----
    (DOC_MAIN, USER_A, 5, "BOOK II — Chapter 5", False,
     "Under cover of a season-storm, Baron Vethran's legions fell upon Far Reach. The "
     "Baron gloated from his command barge as the citadel burned, certain that House "
     "Ardent was finished and the vael was his."),
    (DOC_MAIN, USER_A, 6, "BOOK II — Chapter 6", False,
     "Vethran ordered the bombardment of the Shield Cliffs, sealing the Ardent levies "
     "in the caves to starve. He was alive with triumph, already dividing the spoils of "
     "Kaaran among his lieutenants."),
    (DOC_MAIN, USER_A, 7, "BOOK II — Chapter 7", False,
     "But the harvest crews had not scattered. Lyra gathered the survivors in the deep "
     "sietch and the tide of the war began, slowly, to turn against the invaders from "
     "Giedan."),
    (DOC_MAIN, USER_A, DEATH_CHAPTER, "BOOK II — Chapter 8", False,
     "In the throne room at Far Reach the child Lyra stood before the dais where the "
     "Baron held court. \"You have met the Whisper's edge,\" she said softly, and "
     "stepped back. The old man's breath caught; he sank against the cold stone and "
     "was still, his suspensors drifting his bulk a hand's width from the floor."),
    # ---- DOC_MAIN, reference / appendix (plain statement of the death) ----
    (DOC_MAIN, USER_A, None, "Almanak: The Noble Houses", True,
     "HOUSE VETHRAN. Baron Vethran, called the Iron Baron, died during the Reclamation "
     "of Kaaran; his seat and holdings passed to his nephew Korr Vethran, who ruled but "
     "three years before the line failed."),

    # ---- DOC_OTHER_USER (USER_B) — tenancy ----
    (DOC_OTHER_USER, USER_B, 1, "Field Dispatch", False,
     "Border dispatch, third watch: Baron Vethran's border banners were sighted along "
     "the Giedan frontier, moving toward the salt flats under a grey sky."),

    # ---- DOC_DELETABLE (USER_A) — soft-delete ----
    (DOC_DELETABLE, USER_A, 1, "Provisional Memo", False,
     "The Provisional Accord granted the harvest crews safe passage through the Salt "
     "Gate for the duration of the truce, pending ratification by the council at Far "
     "Reach."),
]


def build_documents() -> List[Document]:
    """Build the corpus as langchain Documents with production-shaped metadata."""
    # total_chunks is per (document_id) like production stamps it within a doc.
    per_doc_total = {}
    for doc_id, *_ in _RECORDS:
        per_doc_total[doc_id] = per_doc_total.get(doc_id, 0) + 1

    per_doc_index = {}
    documents: List[Document] = []
    for doc_id, user_id, chapter_number, chapter_title, is_reference, content in _RECORDS:
        idx = per_doc_index.get(doc_id, 0)
        per_doc_index[doc_id] = idx + 1
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "document_id": doc_id,
                    "user_id": user_id,
                    "document_title": DOC_TITLE if doc_id == DOC_MAIN else f"doc-{doc_id}",
                    "source_type": "text",
                    "chunk_index": idx,
                    "total_chunks": per_doc_total[doc_id],
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "is_reference": is_reference,
                },
            )
        )
    return documents
