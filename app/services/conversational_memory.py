"""
Conversational Memory system for context-aware conversations
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class ConversationSession:
    """
    Represents a single conversation session
    """
    def __init__(self, session_id: str, user_id: Optional[str] = None ):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """
        Adds a message to the conversation session
        """
        message = {
            "role": role,
            "content": content
            "timestamp": datetime.now()
        }
        self.messages.append(message)
        self.last_activity = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this conversation"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": len(self.messages),
        }

class ConversationMemoryManager:
    """
    Manages multiple conversation sessions
    """

    def __init__(self):

