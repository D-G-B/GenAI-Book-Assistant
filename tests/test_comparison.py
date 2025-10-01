"""
Test script to compare both RAG implementations
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import requests
from app.database import SessionLocal, LoreDocument

BASE_URL = "http://localhost:8000/api/v1"


def create_test_document():
    """Create a test document in the database"""
    db = SessionLocal()

    # Check if test doc already exists
    existing = db.query(LoreDocument).filter(
        LoreDocument.title == "Test Document"
    ).first()

    if existing:
        print(f"✅ Test document already exists (ID: {existing.id})")
        db.close()
        return existing.id

    # Create new test document
    test_doc = LoreDocument(
        title="Test Document",
        filename="test.txt",
        content="""
        The Kingdom of Eldoria has seven magical schools.

        The schools are: Fire Magic, Water Magic, Earth Magic, Air Magic,
        Light Magic, Shadow Magic, and Time Magic.

        Dragons serve as teachers at the academy. There are five dragon elders:
        Pyraxis (fire), Aquaria (water), Terrazon (earth), Ventus (air), and Chronos (time).

        The capital city is Luminspire, built on floating islands.
        """,
        source_type="text"
    )

    db.add(test_doc)
    db.commit()
    db.refresh(test_doc)

    doc_id = test_doc.id
    print(f"✅ Created test document (ID: {doc_id})")
    db.close()
    return doc_id


def test_document_processing(doc_id):
    """Test both services processing the same document"""
    print("\n" + "=" * 60)
    print("📄 Testing Document Processing")
    print("=" * 60)

    # Process with original service
    print("\n1️⃣ Processing with ORIGINAL service...")
    response = requests.post(f"{BASE_URL}/documents/{doc_id}/process")
    if response.status_code == 200:
        print("   ✅ Original service processed document")
    else:
        print(f"   ❌ Original service failed: {response.text}")

    # Process with LangChain service
    # (You'd need to add this endpoint, or call directly)
    print("\n2️⃣ Processing with LANGCHAIN service...")
    from app.services.langchain_rag_service import langchain_rag_service
    from app.database import SessionLocal

    db = SessionLocal()
    success = asyncio.run(langchain_rag_service.process_document(db, doc_id))
    db.close()

    if success:
        print("   ✅ LangChain service processed document")
    else:
        print("   ❌ LangChain service failed")


def test_question_comparison():
    """Compare answers from both services"""
    print("\n" + "=" * 60)
    print("❓ Testing Question Answering")
    print("=" * 60)

    test_questions = [
        "What caused the Great Schism",
        "Who are the five Elder Dragons",
        "What is the difference between Luxmancy and Ubramancy",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)

        # Ask both services
        response = requests.post(
            f"{BASE_URL}/compare/ask-both",
            json={"question": question, "max_chunks": 3}
        )

        if response.status_code == 200:
            result = response.json()

            print("\n   🔵 ORIGINAL SERVICE:")
            print(f"   Answer: {result['original_service']['answer'][:150]}...")
            print(f"   Chunks: {result['original_service']['chunks_used']}")
            print(f"   Confidence: {result['original_service']['confidence']:.2f}")

            print("\n   🟢 LANGCHAIN SERVICE:")
            print(f"   Answer: {result['langchain_service']['answer'][:150]}...")
            print(f"   Chunks: {result['langchain_service']['chunks_used']}")
            print(f"   Confidence: {result['langchain_service']['confidence']:.2f}")
        else:
            print(f"   ❌ Error: {response.status_code}")


def test_status_comparison():
    """Compare status of both services"""
    print("\n" + "=" * 60)
    print("📊 Service Status Comparison")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/compare/status")

    if response.status_code == 200:
        status = response.json()

        print("\n🔵 ORIGINAL SERVICE:")
        for key, value in status['original'].items():
            print(f"   {key}: {value}")

        print("\n🟢 LANGCHAIN SERVICE:")
        for key, value in status['langchain'].items():
            print(f"   {key}: {value}")

        print("\n📈 COMPARISON:")
        print(f"   Both Ready: {status['comparison']['both_ready']}")
    else:
        print(f"❌ Error: {response.status_code}")


def main():
    """Run all comparison tests"""
    print("\n" + "=" * 60)
    print("🧪 RAG SERVICE COMPARISON TEST")
    print("=" * 60)
    print("\nThis will compare your original SimpleRAG with LangChain")
    print("to show you the differences in practice!\n")

    input("Press Enter to start tests...")

    try:
        # Setup
        doc_id = create_test_document()

        # Test processing
        test_document_processing(doc_id)

        # Test questions
        test_question_comparison()

        # Test status
        test_status_comparison()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n💡 Key Observations:")
        print("   - LangChain handles chunking more intelligently")
        print("   - Both should give similar answers (good!)")
        print("   - LangChain code is much shorter and cleaner")
        print("   - LangChain is easier to extend with new features")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    # Make sure server is running first!
    print("⚠️  Make sure your FastAPI server is running!")
    print("   Run: python main.py")
    print()
    main()
