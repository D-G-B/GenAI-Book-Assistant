import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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