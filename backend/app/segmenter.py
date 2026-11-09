"""
Incremental sentence segmentation for streaming text.

This is intentionally regex-based rather than a full NLP sentence
tokenizer (spaCy/nltk): it must run on the hot path for every token
batch that arrives from the LLM stream, so it needs to be effectively
free. It trades a small amount of accuracy (abbreviations like "Dr."
can occasionally cause an early split) for near-zero latency.
"""
import re

_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")


class StreamingSentenceSplitter:
    """
    Feed it token fragments as they arrive; it yields complete sentences
    as soon as a sentence boundary appears in the buffer, and keeps the
    remainder buffered for the next call.
    """

    def __init__(self, min_chars: int = 8):
        self._buffer = ""
        self._min_chars = min_chars

    def push(self, fragment: str) -> list[str]:
        self._buffer += fragment
        sentences: list[str] = []

        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not match:
                break
            end = match.end()
            candidate = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]
            if len(candidate) >= self._min_chars:
                sentences.append(candidate)
        return sentences

    def flush(self) -> str | None:
        """Call once the stream ends to get any trailing partial sentence."""
        remainder = self._buffer.strip()
        self._buffer = ""
        if len(remainder) >= self._min_chars:
            return remainder
        return None
