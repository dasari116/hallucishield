"""
Central configuration for HalluciShield.

All knobs that affect real-time latency live here so they can be tuned
without touching pipeline code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    # --- LLM provider (generation) ---
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

    # --- Embeddings (used for retrieval + fallback verification) ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "256"))  # used by mock embedder

    # --- Verification behaviour ---
    # If an OpenAI key is present, verification uses a fast LLM entailment
    # call. Otherwise it falls back to a pure embedding-similarity heuristic
    # so the whole pipeline still runs end-to-end with zero API keys.
    VERIFIER_MODEL: str = os.getenv("VERIFIER_MODEL", "gpt-4o-mini")
    # NOTE: these defaults are tuned for the offline MockEmbedder, which
    # produces coarser similarity scores than real embeddings. If you set
    # OPENAI_API_KEY, consider raising SUPPORTED_THRESHOLD (e.g. 0.75+)
    # since real embedding similarity is much more discriminative.
    SUPPORTED_THRESHOLD: float = float(os.getenv("SUPPORTED_THRESHOLD", "0.45"))
    CONTRADICTED_MARGIN: float = float(os.getenv("CONTRADICTED_MARGIN", "0.12"))

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # --- Real-time pipeline ---
    MIN_SENTENCE_CHARS: int = int(os.getenv("MIN_SENTENCE_CHARS", "8"))
    MAX_CONCURRENT_VERIFICATIONS: int = int(os.getenv("MAX_CONCURRENT_VERIFICATIONS", "6"))
    MOCK_TOKEN_DELAY_MS: int = int(os.getenv("MOCK_TOKEN_DELAY_MS", "18"))

    SAMPLE_DOCS_DIR: str = os.getenv(
        "SAMPLE_DOCS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_docs"),
    )


settings = Settings()
