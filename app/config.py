import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Default Models
    DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL")
    DEFAULT_CLAUDE_MODEL = os.getenv("DEFAULT_CLAUDE_MODEL")
    DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL")

    # App Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

    # Retrieval Settings
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
    LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # Reranker Settings
    # Cross-encoder reranker reorders top-N FAISS candidates for better recall
    # on proper-noun and lexically-mismatched queries. Set RERANKER_ENABLED=False
    # to disable (e.g. low-memory environments).
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "True").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", "30"))

    def validate_api_keys(self):
        """
        Check if the api keys are present
        """
        missing_keys =  []
        if not self.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not self.ANTHROPIC_API_KEY:
            missing_keys.append("ANTHROPIC_API_KEY")
        if not self.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")

        if missing_keys:
            logger.warning("Missing API keys: %s. Add them to your .env to use those models.", ", ".join(missing_keys))

        return len(missing_keys) == 0

# Create global settings instance
settings = Settings()