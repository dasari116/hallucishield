"""
HalluciShield -- real-time RAG-based hallucination detection.

WebSocket pipeline (per connection):

    client sends prompt
        -> stream_generation() yields tokens
        -> StreamingSentenceSplitter emits complete sentences as they form
        -> each sentence spawns an async verify_claim() task immediately
           (generation keeps streaming while verification runs concurrently)
        -> verification results are pushed back to the client as soon as
           they resolve, independent of generation order

This is what makes it "real-time": the user sees hallucination verdicts
trailing the generated text by a few hundred milliseconds, not after the
full response completes.
"""
import asyncio
from contextlib import asynccontextmanager
import glob
import os
import time

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.llm import stream_generation
from app.models import Document
from app.segmenter import StreamingSentenceSplitter
from app.vector_store import vector_store
from app.verifier import verify_claim


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Seed the vector store with the bundled sample knowledge base."""
    paths = sorted(glob.glob(os.path.join(settings.SAMPLE_DOCS_DIR, "*.txt")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = os.path.splitext(os.path.basename(path))[0]
        await vector_store.add_document(Document(doc_id=doc_id, source=doc_id, text=text))
    print(
        f"[HalluciShield] loaded {vector_store.size} chunks from {len(paths)} sample docs "
        f"(mode: {'OpenAI' if settings.OPENAI_API_KEY else 'offline/mock'})",
        flush=True,
    )
    yield


app = FastAPI(title="HalluciShield", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "indexed_chunks": vector_store.size,
        "mode": "openai" if settings.OPENAI_API_KEY else "offline",
    }


@app.get("/api/sources")
async def sources():
    return {"sources": vector_store.sources, "total_chunks": vector_store.size}


@app.post("/api/ingest")
async def ingest(doc: Document):
    n_chunks = await vector_store.add_document(doc)
    return {"doc_id": doc.doc_id, "chunks_added": n_chunks, "total_chunks": vector_store.size}


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    allowed_types = {".txt", ".md", ".markdown", ".csv"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed_types:
        raise HTTPException(status_code=400, detail="Only .txt, .md, .markdown, and .csv files are supported.")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Files must be smaller than 5 MB.")
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The file must be UTF-8 encoded text.") from exc
    if not text:
        raise HTTPException(status_code=400, detail="The selected file is empty.")

    doc_id = os.path.splitext(os.path.basename(file.filename or "upload"))[0]
    n_chunks = await vector_store.add_document(Document(doc_id=doc_id, source=file.filename or doc_id, text=text))
    return {"doc_id": doc_id, "source": file.filename, "chunks_added": n_chunks, "total_chunks": vector_store.size}


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            prompt = await ws.receive_text()
            await _run_pipeline(ws, prompt)
    except WebSocketDisconnect:
        pass


async def _run_pipeline(ws: WebSocket, prompt: str):
    splitter = StreamingSentenceSplitter(min_chars=settings.MIN_SENTENCE_CHARS)
    sentence_id = 0
    verification_tasks: list[asyncio.Task] = []
    start = time.perf_counter()

    await ws.send_json({"type": "start"})

    async def dispatch_when_ready(task: asyncio.Task):
        result = await task
        await ws.send_json({"type": "verdict", **result.model_dump()})

    async for token in stream_generation(prompt):
        await ws.send_json({"type": "token", "text": token})

        for sentence in splitter.push(token):
            sid = sentence_id
            sentence_id += 1
            await ws.send_json({"type": "sentence", "sentence_id": sid, "text": sentence})
            task = asyncio.create_task(verify_claim(sid, sentence))
            verification_tasks.append(asyncio.create_task(dispatch_when_ready(task)))

    trailing = splitter.flush()
    if trailing:
        sid = sentence_id
        await ws.send_json({"type": "sentence", "sentence_id": sid, "text": trailing})
        task = asyncio.create_task(verify_claim(sid, trailing))
        verification_tasks.append(asyncio.create_task(dispatch_when_ready(task)))

    if verification_tasks:
        await asyncio.gather(*verification_tasks)

    await ws.send_json({
        "type": "done",
        "total_latency_ms": round((time.perf_counter() - start) * 1000, 1),
    })
