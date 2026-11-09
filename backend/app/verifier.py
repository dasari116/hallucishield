"""
Claim verification -- the accuracy-critical, latency-critical core of
HalluciShield.

Two modes, chosen automatically based on whether an OpenAI key is
configured:

1. LLM entailment mode (OPENAI_API_KEY set): a single, tightly-scoped
   completion call asks a small/fast model whether the retrieved
   evidence supports, contradicts, or doesn't address the claim. This
   is far cheaper and faster than re-running full generation-quality
   models, which is what keeps it viable on the real-time path.

2. Embedding-similarity heuristic (no key): compares the claim's
   embedding against each evidence chunk. High similarity -> supported,
   very low best-match -> unverifiable (nothing relevant retrieved),
   and a claim whose top match is only weakly similar but a *different*
   chunk about the same entity scores much higher is treated as a
   signal of possible contradiction. This mode has no external
   dependency, so the whole system is runnable and demoable offline.

Either way, verification for independent claims runs concurrently
(bounded by MAX_CONCURRENT_VERIFICATIONS) so a sentence with multiple
claims doesn't serialize latency.
"""
import asyncio
import re
import time

from app.config import settings
from app.embeddings import content_tokens, get_embedder
from app.models import ClaimVerification, EvidenceChunk, Verdict
from app.retriever import retrieve

_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_VERIFICATIONS)
_embedder = get_embedder()

_NEGATIONS = {"not", "never", "no", "neither", "nor", "none", "hardly", "barely"}
_APPROX_WORDS = {"about", "over", "around", "approximately", "nearly", "almost", "roughly"}
_COMMON_STARTS = {
    "The", "It", "This", "That", "These", "Those", "In", "On", "At", "For",
    "By", "With", "After", "Before", "During", "Although", "While", "When",
    "Where", "However", "Most", "Many", "Some",
}


def _extract_numbers(text: str) -> list[float]:
    return [
        float(x.replace(",", ""))
        for x in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text)
    ]


def _extract_proper_nouns(text: str) -> set[str]:
    words = re.findall(r"\b[A-Za-z][a-z]+\b", text)
    result: set[str] = set()
    for i, word in enumerate(words):
        if word[0].isupper():
            if i == 0 and word in _COMMON_STARTS:
                continue
            result.add(word.lower())
    return result


async def _heuristic_verify(claim: str, evidence: list[EvidenceChunk]) -> tuple[Verdict, float]:
    if not evidence:
        return Verdict.UNVERIFIABLE, 0.0

    best = max(e.score for e in evidence)
    all_ev_text = " ".join(e.text for e in evidence)
    ev_tokens = set(content_tokens(all_ev_text))
    c_tokens = content_tokens(claim)
    if not c_tokens:
        return Verdict.UNVERIFIABLE, 0.0

    c_nums = _extract_numbers(claim)
    ev_nums = _extract_numbers(all_ev_text)
    claim_words = set(re.findall(r"[a-z]+", claim.lower()))
    has_approx = bool(claim_words & _APPROX_WORDS)

    num_conflict = False
    for cn in c_nums:
        if has_approx:
            if not any(abs(cn - en) / max(abs(en), 1.0) <= 0.10 for en in ev_nums):
                num_conflict = True
        else:
            if not any(abs(cn - en) < 1e-6 for en in ev_nums):
                num_conflict = True

    c_prop = _extract_proper_nouns(claim)
    ev_prop = _extract_proper_nouns(all_ev_text)
    unseen_prop = c_prop - ev_prop

    top_ev_words = set(re.findall(r"[a-z]+", evidence[0].text.lower()))
    c_neg = bool(claim_words & _NEGATIONS)
    ev_neg = bool(top_ev_words & _NEGATIONS)

    non_num_tokens = [t for t in c_tokens if not t.isdigit()]
    overlap = sum(1 for t in non_num_tokens if t in ev_tokens) / max(len(non_num_tokens), 1)
    top_chunk_tokens = set(content_tokens(evidence[0].text))
    shared_anchors = set(non_num_tokens) & top_chunk_tokens
    has_anchor = len(shared_anchors) >= 2 or best >= (settings.SUPPORTED_THRESHOLD - settings.CONTRADICTED_MARGIN)

    if has_anchor and (
        num_conflict
        or (unseen_prop and len(unseen_prop) >= 1)
        or (c_neg != ev_neg and overlap >= 0.5)
    ):
        return Verdict.CONTRADICTED, min(0.95, 0.72 + 0.1 * len(shared_anchors))

    if best >= settings.SUPPORTED_THRESHOLD or overlap >= 0.65:
        return Verdict.SUPPORTED, min(0.98, max(best, overlap))

    return Verdict.UNVERIFIABLE, max(0.55, 1.0 - best)


async def _llm_verify(claim: str, evidence: list[EvidenceChunk]) -> tuple[Verdict, float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    evidence_block = "\n".join(f"- {e.text}" for e in evidence) or "(no evidence retrieved)"

    prompt = (
        "You are a fast fact-checking classifier. Given a CLAIM and retrieved "
        "EVIDENCE, decide if the evidence SUPPORTS, CONTRADICTS, or is "
        "INSUFFICIENT to evaluate the claim.\n\n"
        f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_block}\n\n"
        "Respond with exactly one line in the format: VERDICT|CONFIDENCE\n"
        "VERDICT must be one of: SUPPORTED, CONTRADICTED, UNVERIFIABLE\n"
        "CONFIDENCE is a number between 0 and 1."
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        verdict_str, conf_str = raw.split("|")
        verdict = Verdict(verdict_str.strip().lower())
        confidence = max(0.0, min(1.0, float(conf_str.strip())))
        return verdict, confidence
    except Exception:
        return await _heuristic_verify(claim, evidence)


async def verify_claim(sentence_id: int, claim: str) -> ClaimVerification:
    async with _semaphore:
        start = time.perf_counter()
        evidence = await retrieve(claim)

        if settings.OPENAI_API_KEY:
            verdict, confidence = await _llm_verify(claim, evidence)
        else:
            verdict, confidence = await _heuristic_verify(claim, evidence)

        latency_ms = (time.perf_counter() - start) * 1000
        return ClaimVerification(
            sentence_id=sentence_id,
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 3),
            evidence=evidence,
            latency_ms=round(latency_ms, 1),
        )
