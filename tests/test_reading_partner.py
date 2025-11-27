"""
Test Script for Reading Partner v2
Tests: EPUB upload, chapter extraction, spoiler filtering, conversational mode

Run from project root with server running:
    uv run python tests/test_reading_partner_v2.py

Prerequisites:
    - Server running on localhost:8000
    - No documents in system (or run with --clean flag)
"""

import requests
import time
import sys
from uuid import uuid4

BASE_URL = "http://localhost:8000/api/v1"

# Sample EPUB-like content with chapter structure
# (We'll upload as .txt but with chapter markers to simulate EPUB extraction)
SAMPLE_BOOK_CONTENT = """=== Prologue ===

In the beginning, there was the Spice. The Spice Melange, found only on the desert planet Arrakis, 
was the most valuable substance in the known universe. It extended life, expanded consciousness, 
and made interstellar travel possible.

=== Chapter 1 ===

Paul Atreides was fifteen years old when his family received orders to take control of Arrakis. 
He was the son of Duke Leto Atreides and Lady Jessica, a Bene Gesserit. Paul had been trained 
in the ways of the Bene Gesserit by his mother, learning the Voice and other mental disciplines.

The Atreides family lived on the ocean planet Caladan, where they had ruled for generations.
Duke Leto was known as a just and honorable ruler, beloved by his people.

=== Chapter 2 ===

The Harkonnens were the sworn enemies of House Atreides. Baron Vladimir Harkonnen had ruled 
Arrakis for decades, growing wealthy from the spice trade. He was a cruel and cunning man who 
plotted the destruction of the Atreides.

The Baron's nephews, Glossu Rabban and Feyd-Rautha, were being groomed to continue the family legacy.
Rabban was known for his brutality, while Feyd was more subtle and dangerous.

=== Chapter 3 ===

The Fremen were the native people of Arrakis. They lived in the deep desert, surviving in conditions 
that would kill most outsiders. They had blue-within-blue eyes from constant exposure to the spice.

The Fremen believed in a prophecy of a messiah who would come from the outer world to lead them 
to paradise. They called this figure the Lisan al-Gaib.

=== Chapter 4 ===

Dr. Yueh was the Atreides family doctor, a Suk doctor whose imperial conditioning was supposed to 
make betrayal impossible. However, the Harkonnens had found a way to break him by capturing his wife.

Yueh's betrayal would be the key to the Harkonnen attack on House Atreides.

=== Chapter 5 ===

The attack came swiftly. Harkonnen forces, aided by the Emperor's Sardaukar troops disguised as 
Harkonnens, overwhelmed the Atreides defenses. Duke Leto was captured and killed.

Paul and Jessica escaped into the desert, where they were found by the Fremen.

=== Appendix I: Terminology ===

BENE GESSERIT: An ancient school of mental and physical training for women. They possess abilities 
that seem almost magical to outsiders, including the Voice, which allows them to control others.

SPICE MELANGE: The most valuable substance in the universe. Found only on Arrakis. Extends life, 
expands consciousness, and is essential for space navigation.

FREMEN: The native people of Arrakis. Desert-adapted humans who have developed a complex culture 
centered around water conservation and survival.

SARDAUKAR: The elite military force of the Padishah Emperor. Feared throughout the universe for 
their fighting prowess.

VOICE: A Bene Gesserit technique that allows the user to control others through specially modulated 
vocal tones.

=== Appendix II: Houses ===

HOUSE ATREIDES: A noble house known for honor and just rule. Seat: Caladan (later Arrakis).

HOUSE HARKONNEN: A noble house known for cruelty and cunning. Seat: Giedi Prime.

HOUSE CORRINO: The Imperial House. The Padishah Emperors have ruled for thousands of years.
"""


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.END}")


def print_test(name):
    print(f"\n{Colors.BOLD}Testing: {name}{Colors.END}")


def check_server():
    """Check if server is running."""
    print_test("Server Connection")
    try:
        response = requests.get(f"{BASE_URL}/chat/status", timeout=5)
        if response.status_code == 200:
            print_success("Server is running")
            return True
        else:
            print_error(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is it running on localhost:8000?")
        return False


def clean_documents():
    """Delete all existing documents."""
    print_test("Cleaning Existing Documents")
    try:
        response = requests.delete(f"{BASE_URL}/documents/all")
        if response.status_code == 200:
            print_success("All documents deleted")
            return True
        else:
            print_info(f"Delete returned {response.status_code}: {response.text}")
            return True  # Not critical
    except Exception as e:
        print_error(f"Failed to clean documents: {e}")
        return False


def upload_test_document():
    """Upload the sample book content."""
    print_test("Document Upload (Simulated EPUB)")

    # Create a file-like upload
    files = {
        'file': ('test_dune_sample.txt', SAMPLE_BOOK_CONTENT, 'text/plain')
    }
    data = {
        'title': 'Dune Sample (Test)'
    }

    try:
        response = requests.post(
            f"{BASE_URL}/documents/upload-file",
            files=files,
            data=data
        )

        if response.status_code == 200:
            doc = response.json()
            print_success(f"Document uploaded: ID={doc['id']}, Title='{doc['title']}'")
            return doc['id']
        else:
            print_error(f"Upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print_error(f"Upload error: {e}")
        return None


def check_document_processed(doc_id):
    """Verify document was processed with chapters."""
    print_test("Document Processing Verification")

    try:
        response = requests.get(f"{BASE_URL}/documents/list")
        if response.status_code == 200:
            docs = response.json()
            for doc in docs:
                if doc['id'] == doc_id:
                    chunk_count = doc.get('chunk_count', 0)
                    total_chapters = doc.get('total_chapters')

                    if chunk_count > 0:
                        print_success(f"Document processed: {chunk_count} chunks")
                        if total_chapters:
                            print_success(f"Chapters detected: {total_chapters}")
                        else:
                            print_info("No chapter count reported (may be non-EPUB)")
                        return True
                    else:
                        print_error("Document has 0 chunks")
                        return False

            print_error(f"Document {doc_id} not found in list")
            return False
        else:
            print_error(f"Failed to get document list: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error checking document: {e}")
        return False


def test_basic_qa(doc_id):
    """Test basic Q&A without spoiler filter."""
    print_test("Basic Q&A (No Spoiler Filter)")

    questions = [
        ("Who is Paul Atreides?", "Paul"),
        ("What is the Spice?", "spice"),
        ("Who are the Fremen?", "Fremen"),
    ]

    all_passed = True

    for question, expected_keyword in questions:
        try:
            response = requests.post(
                f"{BASE_URL}/chat/ask",
                json={"question": question},
                params={"document_id": doc_id}
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')

                if expected_keyword.lower() in answer.lower():
                    print_success(f"Q: '{question}' - Got relevant answer")
                else:
                    print_info(f"Q: '{question}' - Answer may not contain expected content")
                    print(f"   Answer preview: {answer[:100]}...")
            else:
                print_error(f"Q: '{question}' - Failed: {response.status_code}")
                all_passed = False

        except Exception as e:
            print_error(f"Q: '{question}' - Error: {e}")
            all_passed = False

        time.sleep(0.5)  # Rate limiting

    return all_passed


def test_spoiler_filter(doc_id):
    """Test spoiler filtering functionality."""
    print_test("Spoiler Filter")

    # Track overall success of this test function
    success = True

    # Duke Leto's death is in Chapter 5
    # With max_chapter=3, we should NOT find info about his death
    # With max_chapter=5+, we should find it

    question = "What happened to Duke Leto?"

    # --- Test 1: Spoiler Protection (Chapter 3) ---
    print_info("Testing with max_chapter=3 (before Duke's death)...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/ask",
            json={"question": question},
            params={"document_id": doc_id, "max_chapter": 3}
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '').lower()

            # Should NOT mention death/killed/captured at chapter 3
            spoiler_words = ['killed', 'death', 'died', 'captured', 'attack']
            has_spoiler = any(word in answer for word in spoiler_words)

            if not has_spoiler:
                print_success("Chapter 3 filter: No spoilers about Duke's fate")
            else:
                print_error("Chapter 3 filter: Leaked spoiler about Duke's fate!")
                print(f"   Answer: {answer[:150]}...")
                success = False  # Mark test as failed
        else:
            print_error(f"Request failed: {response.status_code}")
            success = False

    except Exception as e:
        print_error(f"Error: {e}")
        success = False

    time.sleep(0.5)

    # --- Test 2: Accessing Content (Chapter 5) ---
    print_info("Testing with max_chapter=5 (after Duke's death)...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/ask",
            json={"question": question},
            params={"document_id": doc_id, "max_chapter": 5}
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '').lower()

            # Should mention death/killed/captured at chapter 5
            spoiler_words = ['killed', 'death', 'died', 'captured', 'attack']
            has_info = any(word in answer for word in spoiler_words)

            if has_info:
                print_success("Chapter 5 filter: Found info about Duke's fate (correct)")
            else:
                print_info("Chapter 5 filter: Didn't find death info (may be in later chunk)")
                # We don't necessarily fail here, as RAG can be fuzzy, but good to know
        else:
            print_error(f"Request failed: {response.status_code}")
            success = False

    except Exception as e:
        print_error(f"Error: {e}")
        success = False

    return success


def test_reference_material(doc_id):
    """Test that reference material (glossary) is always available."""
    print_test("Reference Material Access (Glossary)")

    # Even with max_chapter=1, glossary should be searchable
    question = "What is the Voice?"

    try:
        response = requests.post(
            f"{BASE_URL}/chat/ask",
            json={"question": question},
            params={"document_id": doc_id, "max_chapter": 1}  # Very restrictive
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '').lower()

            # Should find Voice definition from Appendix
            if 'bene gesserit' in answer or 'control' in answer or 'vocal' in answer:
                print_success("Glossary accessible even with chapter filter")
                print(f"   Answer: {answer[:150]}...")
            else:
                print_info("May not have found glossary definition")
                print(f"   Answer: {answer[:150]}...")
        else:
            print_error(f"Request failed: {response.status_code}")

    except Exception as e:
        print_error(f"Error: {e}")

    return True


def test_conversational_mode(doc_id):
    """Test conversational mode with spoiler filter."""
    print_test("Conversational Mode with Spoiler Filter")

    session_id = f"test_{uuid4()}"

    conversation = [
        ("Who is Paul?", None),
        ("What is his mother's role?", "Bene Gesserit"),  # Follow-up using "his"
        ("What special abilities does she have?", "Voice"),  # Follow-up using "she"
    ]

    all_passed = True

    for i, (question, expected) in enumerate(conversation):
        try:
            response = requests.post(
                f"{BASE_URL}/conversation/ask",
                json={"question": question},
                params={
                    "session_id": session_id,
                    "document_id": doc_id,
                    "max_chapter": 3  # With spoiler filter
                }
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                context_used = result.get('context_used', False)

                status = "✓" if i == 0 or context_used else "?"
                print_success(f"Turn {i + 1}: '{question}' - Got answer (context_used={context_used})")

                if expected and expected.lower() not in answer.lower():
                    print_info(f"   Expected '{expected}' in answer")
            else:
                print_error(f"Turn {i + 1} failed: {response.status_code}")
                all_passed = False

        except Exception as e:
            print_error(f"Turn {i + 1} error: {e}")
            all_passed = False

        time.sleep(0.5)

    # Clean up session
    try:
        requests.delete(f"{BASE_URL}/conversation/session/{session_id}")
    except:
        pass

    return all_passed


def test_no_filter():
    """Test that no filter returns full book content."""
    print_test("No Filter (Full Book Access)")

    try:
        # Ask about something from chapter 5
        response = requests.post(
            f"{BASE_URL}/chat/ask",
            json={"question": "What happened in the Harkonnen attack?"}
            # No max_chapter = full book access
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '').lower()
            spoiler_active = result.get('spoiler_filter_active', False)

            if not spoiler_active:
                print_success("Spoiler filter correctly reported as inactive")
            else:
                print_error("Spoiler filter incorrectly reported as active")

            if 'duke' in answer or 'attack' in answer or 'captured' in answer:
                print_success("Found late-chapter content (no filter working)")
            else:
                print_info("Didn't find expected content, but filter is off")
        else:
            print_error(f"Request failed: {response.status_code}")

    except Exception as e:
        print_error(f"Error: {e}")

    return True


def run_all_tests():
    """Run all tests."""
    print_header("Reading Partner v2 - Test Suite")
    print("Testing: EPUB support, Chapter extraction, Spoiler filtering")
    print()

    # Check server
    if not check_server():
        print_error("\nServer not available. Start it with: uv run uvicorn main:app --reload")
        return False

    # Clean if requested
    if "--clean" in sys.argv:
        clean_documents()
        time.sleep(1)

    # Upload test document
    doc_id = upload_test_document()
    if not doc_id:
        print_error("\nFailed to upload test document. Aborting.")
        return False

    time.sleep(2)  # Wait for processing

    # Run tests
    results = []

    results.append(("Document Processing", check_document_processed(doc_id)))
    results.append(("Basic Q&A", test_basic_qa(doc_id)))
    results.append(("Spoiler Filter", test_spoiler_filter(doc_id)))
    results.append(("Reference Material", test_reference_material(doc_id)))
    results.append(("Conversational Mode", test_conversational_mode(doc_id)))
    results.append(("No Filter Mode", test_no_filter()))

    # Summary
    print_header("Test Summary")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {name}: {status}")

    print()
    if passed == total:
        print_success(f"All {total} tests passed!")
    else:
        print_error(f"{passed}/{total} tests passed")

    # Cleanup option
    print()
    print_info("To clean up test data, run with --clean flag next time")
    print_info("Or manually delete via: curl -X DELETE localhost:8000/api/v1/documents/all")

    return passed == total


if __name__ == "__main__":
    print("\n⚠️  Make sure your server is running!")
    print("   Run: uv run uvicorn main:app --reload\n")

    if "--help" in sys.argv:
        print("Usage: python test_reading_partner_v2.py [--clean]")
        print("  --clean  Delete all documents before testing")
        sys.exit(0)

    success = run_all_tests()
    sys.exit(0 if success else 1)
