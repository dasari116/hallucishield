"""Thin retrieval interface sitting in front of the vector store."""
from app.config import settings
from app.models import EvidenceChunk
from app.vector_store import vector_store


async def retrieve(claim: str, top_k: int | None = None) -> list[EvidenceChunk]:
    return await vector_store.search(claim, top_k or settings.TOP_K)
