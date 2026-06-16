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

    # Auth / Session (Phase 2 Increment 2 — Google OAuth + signed-cookie session)
    # SESSION_SECRET_KEY signs the session cookie; MUST be set to a real random
    # value in any non-local deployment (generate: python -c "import secrets; print(secrets.token_hex(32))").
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-insecure-change-me")
    # Add the Secure flag to the session cookie. Leave False for local http dev;
    # set True wherever the app is served over HTTPS so the cookie can't be sent
    # in cleartext (Starlette only emits Secure when https_only=True).
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False").lower() == "true"
    GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    # When True and no session user is present, get_current_user falls back to the
    # fixed dev user instead of returning 401. Lets local dev / tests run without
    # Google credentials. NEVER enable in a real deployment.
    DEV_AUTH_BYPASS = os.getenv("DEV_AUTH_BYPASS", "False").lower() == "true"

    # App Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

    # Input Limits
    # Reject oversized uploads before loading them fully into memory (OOM guard),
    # and cap question length to avoid unbounded token use on a single query.
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

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

    # Chapter Detection Settings
    # The hybrid detector (regex anchors + one LLM labelling call) runs once at
    # ingest to number chapters in story order and flag front/back matter — things
    # the pure-regex detector can't do. Set LLM_CHAPTER_DETECTION_ENABLED=False to
    # skip it and use the regex detector only. The labelling call emits one JSON
    # object per section, which needs more output room than a chat answer, so it
    # gets its own token cap instead of the chat-sized MAX_TOKENS.
    LLM_CHAPTER_DETECTION_ENABLED = os.getenv("LLM_CHAPTER_DETECTION_ENABLED", "True").lower() == "true"
    # Default 16000: the first-fallback model gemini-2.5-flash is a "thinking"
    # model whose reasoning consumes the output-token budget, so the labelling
    # JSON for a full book truncates below ~16k (verified 1.00 on Dune at 16000).
    LLM_CHAPTER_DETECTION_MAX_TOKENS = int(os.getenv("LLM_CHAPTER_DETECTION_MAX_TOKENS", "16000"))

    def validate_api_keys(self):
        """
        Warn about missing keys and key/model mismatches at startup.

        Returns True only if all three provider keys are present. Also warns when a
        key is set without its matching DEFAULT_*_MODEL: `_initialize_llms` silently
        skips such a provider, so without this the gap only shows up as missing
        capacity (or a query-time failure if it leaves no provider configured).
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

        incomplete = [
            f"{key_name} is set but {model_name} is missing"
            for key, model, key_name, model_name in (
                (self.OPENAI_API_KEY, self.DEFAULT_OPENAI_MODEL, "OPENAI_API_KEY", "DEFAULT_OPENAI_MODEL"),
                (self.ANTHROPIC_API_KEY, self.DEFAULT_CLAUDE_MODEL, "ANTHROPIC_API_KEY", "DEFAULT_CLAUDE_MODEL"),
                (self.GOOGLE_API_KEY, self.DEFAULT_GEMINI_MODEL, "GOOGLE_API_KEY", "DEFAULT_GEMINI_MODEL"),
            )
            if key and not model
        ]
        if incomplete:
            logger.warning(
                "Incomplete provider config (these providers will be unavailable): %s",
                "; ".join(incomplete),
            )

        return len(missing_keys) == 0

# Create global settings instance
settings = Settings()