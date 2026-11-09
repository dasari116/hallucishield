"""
In-memory vector store with cosine-similarity search.

Chosen over FAISS/pgvector for this reference implementation so the
project runs anywhere with zero native-dependency install pain. The
interface (add / search) is the seam to swap in FAISS (IndexHNSWFlat)
or pgvector for production-scale, lower-latency retrieval -- see
README "Scaling to production".
"""
import re
import numpy as np

from app.config import settings
from app.embeddings import get_embedder
from app.models import Document, EvidenceChunk


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    words = cleaned.split()
    if len(sentences) >= 2 and len(words) <= size:
        chunks = [
            " ".join(sentences[i : i + 2])
            for i in range(0, len(sentences), 1)
            if " ".join(sentences[i : i + 2])
        ]
        if len(sentences) > 2:
            chunks = chunks[:-1]
        return chunks

    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


class VectorStore:
    def __init__(self):
        self._embedder = get_embedder()
        self._vectors: np.ndarray | None = None
        self._chunks: list[EvidenceChunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[dict[str, int | str]]:
        counts: dict[str, dict[str, int | str]] = {}
        for chunk in self._chunks:
            entry = counts.setdefault(chunk.source, {"source": chunk.source, "doc_id": chunk.doc_id, "chunks": 0})
            entry["chunks"] = int(entry["chunks"]) + 1
        return list(counts.values())

    async def add_document(self, doc: Document) -> int:
        pieces = _chunk_text(doc.text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not pieces:
            return 0
        new_vecs = await self._embedder.embed(pieces)
        for piece in pieces:
            self._chunks.append(
                EvidenceChunk(doc_id=doc.doc_id, source=doc.source, text=piece, score=0.0)
            )
        self._vectors = new_vecs if self._vectors is None else np.vstack([self._vectors, new_vecs])
        return len(pieces)

    async def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        query_vec = (await self._embedder.embed([query]))[0]
        scores = self._vectors @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            chunk = self._chunks[int(idx)]
            results.append(
                EvidenceChunk(
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=float(scores[idx]),
                )
            )
        return results


# Process-wide singleton so the index stays warm between requests --
# this is what keeps retrieval latency low in the real-time path.
vector_store = VectorStore()
