"""
Test conversational memory feature
Run: .venv/bin/python3 tests/test_conversational.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
from uuid import uuid4

BASE_URL = "http://localhost:8000/api/v1"


def test_conversation():
    """Test a multi-turn conversation."""

    # Generate a unique session ID
    session_id = f"test_{uuid4()}"

    print("=" * 60)
    print("💬 Testing Conversational Memory")
    print("=" * 60)
    print(f"Session ID: {session_id}\n")

    # Conversation flow that tests context
    conversation = [
        ("What are the seven schools of magic?", "Should list all seven schools"),
        ("Which one does Pyraxis teach?", "Should understand 'one' refers to schools"),
        ("What's his element?", "Should understand 'his' refers to Pyraxis"),
        ("Are there other dragon teachers?", "Should reference previous context"),
    ]

    for i, (question, expectation) in enumerate(conversation, 1):
        print(f"\n{'=' * 60}")
        print(f"Turn {i}: {question}")
        print(f"Expected: {expectation}")
        print("-" * 60)

        try:
            response = requests.post(
                f"{BASE_URL}/conversation/ask",
                params={"session_id": session_id},
                json={"question": question},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Answer: {result['answer'][:200]}...")
                print(f"📊 Context used: {result.get('context_used', False)}")
                print(f"📝 Conversation length: {result.get('conversation_length', 0)} messages")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        except Exception as e:
            print(f"❌ Request failed: {e}")
            break

        time.sleep(1)  # Small delay between questions

    # Get conversation history
    print(f"\n{'=' * 60}")
    print("📚 Retrieving Conversation History")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/conversation/history/{session_id}")
        if response.status_code == 200:
            history = response.json()
            print(f"\nTotal messages: {history['message_count']}")
            for i, msg in enumerate(history['history'], 1):
                role = "👤 User" if msg['role'] == 'human' else "🤖 Assistant"
                print(f"\n{i}. {role}: {msg['content'][:100]}...")
        else:
            print(f"❌ Failed to get history: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to get history: {e}")

    # Clean up
    print(f"\n{'=' * 60}")
    print("🧹 Cleaning up")
    print("=" * 60)
    try:
        response = requests.delete(f"{BASE_URL}/conversation/session/{session_id}")
        if response.status_code == 200:
            print("✅ Session cleared successfully")
        else:
            print(f"⚠️ Failed to clear session: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Failed to clear session: {e}")

    print("\n" + "=" * 60)
    print("✅ Conversational Memory Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️  Make sure your server is running!")
    print("   Run: uvicorn main:app --reload\n")

    input("Press Enter to start test...")
    test_conversation()