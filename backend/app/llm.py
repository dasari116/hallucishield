"""
Streaming generation layer.

Uses OpenAI's streaming chat completion when OPENAI_API_KEY is set.
Otherwise falls back to a mock token generator, so the real-time
pipeline (segmentation -> retrieval -> verification -> WebSocket push)
can be exercised and demoed with zero API keys.
"""
import asyncio
import re
from collections.abc import AsyncIterator

from app.config import settings
from app.vector_store import vector_store

_MOCK_RESPONSE = (
    "The Eiffel Tower was completed in 1889 and stands about 330 meters tall. "
    "It was originally intended as a temporary structure for the World's Fair. "
    "The tower is located in Berlin, Germany, near the river Spree. "
    "Gustave Eiffel's company designed and built the structure. "
    "It remained the tallest man-made structure in the world for over 40 years."
)


async def _build_offline_response(prompt: str) -> str:
    lower = prompt.lower()
    if "eiffel" in lower or not lower.strip():
        return _MOCK_RESPONSE

    hits = await vector_store.search(prompt, top_k=2)
    if hits and hits[0].score >= 0.15:
        top_text = hits[0].text.strip()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 12]
        if sents:
            first = sents[0] if sents[0].endswith((".", "!", "?")) else sents[0] + "."
            second = (
                (sents[1] if sents[1].endswith((".", "!", "?")) else sents[1] + ".")
                if len(sents) > 1
                else "The Eiffel Tower was completed in 1889 and stands about 330 meters tall."
            )
            mutated = re.sub(r"\b\d{4}\b", "1974", first, count=1)
            if mutated == first:
                mutated = "The tower is located in Berlin, Germany, near the river Spree."
            unverifiable = "Independent analysts estimate annual maintenance requires over ninety specialized contractors."
            return f"{first} {unverifiable} {mutated} {second}"

    return _MOCK_RESPONSE


async def _mock_stream(prompt: str) -> AsyncIterator[str]:
    response_text = await _build_offline_response(prompt)
    for word in response_text.split(" "):
        yield word + " "
        await asyncio.sleep(settings.MOCK_TOKEN_DELAY_MS / 1000)


async def _openai_stream(prompt: str) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        stream = await client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        async for token in _mock_stream(prompt):
            yield token


def stream_generation(prompt: str) -> AsyncIterator[str]:
    if settings.OPENAI_API_KEY:
        return _openai_stream(prompt)
    return _mock_stream(prompt)
