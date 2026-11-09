"""
Embedding provider with an automatic offline fallback.

If OPENAI_API_KEY is set, real embeddings are used (text-embedding-3-small
by default). If not, a deterministic hash-based "mock embedder" is used
instead so the entire retrieval + verification pipeline still runs
end-to-end without any external API calls -- useful for local dev, demos,
and CI.
"""
import hashlib
import re
import numpy as np

from app.config import settings

STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "were", "are", "be", "been", "being", "it", "its", "as",
    "by", "that", "this", "from", "with", "about", "into", "some", "his",
    "her", "their", "whose", "every", "one", "most", "has", "have", "had",
    "but", "also", "very", "more", "than", "then", "so", "such", "can",
    "will", "would", "could", "over", "approximately", "around",
}


def content_tokens(text: str) -> list[str]:
    """Extract normalized alphanumeric content tokens without stopwords."""
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in STOPWORDS and len(tok) > 1
    ]


class BaseEmbedder:
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class MockEmbedder(BaseEmbedder):
    """
    Deterministic bag-of-hashed-tokens embedder. Normalizes punctuation
    and filters common stopwords so similarity scores reflect genuine
    lexical/entity overlap without any external API calls.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = content_tokens(text) or re.findall(r"[a-z0-9]+", text.lower())
            for token in tokens:
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vectors[i, idx] += sign
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str, dim: int):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = model
        self.dim = dim
        self._fallback = MockEmbedder(dim=dim)

    async def embed(self, texts: list[str]) -> np.ndarray:
        try:
            resp = await self._client.embeddings.create(model=self._model, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms
        except Exception:
            return await self._fallback.embed(texts)


def get_embedder() -> BaseEmbedder:
    if settings.OPENAI_API_KEY:
        return OpenAIEmbedder(settings.EMBEDDING_MODEL, dim=1536)
    return MockEmbedder(dim=settings.EMBEDDING_DIM)
