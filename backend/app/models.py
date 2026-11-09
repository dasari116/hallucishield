"""Pydantic schemas shared across the pipeline and the WebSocket API."""
from enum import Enum
from pydantic import BaseModel


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class EvidenceChunk(BaseModel):
    doc_id: str
    source: str
    text: str
    score: float


class ClaimVerification(BaseModel):
    sentence_id: int
    claim: str
    verdict: Verdict
    confidence: float
    evidence: list[EvidenceChunk] = []
    latency_ms: float


class WSIncoming(BaseModel):
    prompt: str


class Document(BaseModel):
    doc_id: str
    source: str
    text: str
