# HalluciShield

**Real-time RAG-based system for AI hallucination detection and mitigation.**

Instead of generating a full LLM response and checking it afterward, HalluciShield
verifies claims **as the response streams**, so hallucination verdicts appear
only a few hundred milliseconds behind the visible text.

## How it works

```
prompt
  │
  ▼
LLM token stream (OpenAI, or built-in mock if no API key)
  │
  ▼
Streaming sentence segmentation (fires as soon as a sentence completes)
  │
  ▼
For each sentence, concurrently:
  ├── Retrieve top-k evidence chunks (in-memory vector search)
  └── Verify claim against evidence
        - LLM entailment call if OPENAI_API_KEY is set
        - Embedding-similarity heuristic otherwise (fully offline)
  │
  ▼
WebSocket pushes {token}, {sentence}, {verdict} events to the client
as soon as each is ready — verification never blocks generation.
```

The whole system runs **with zero API keys** out of the box (mock LLM +
mock embedder), so you can see the real-time pipeline mechanics
immediately. Set `OPENAI_API_KEY` to get real generation and much more
accurate verification.

## Project structure

```
hallucishield/
├── backend/            FastAPI app, WebSocket pipeline, RAG engine
│   ├── app/
│   │   ├── main.py          WebSocket endpoint + pipeline orchestration
│   │   ├── llm.py           streaming generation (OpenAI / mock)
│   │   ├── segmenter.py     incremental sentence splitting
│   │   ├── retriever.py     retrieval interface
│   │   ├── verifier.py      claim verification (LLM entailment / heuristic)
│   │   ├── vector_store.py  in-memory vector index
│   │   ├── embeddings.py    embedding provider (OpenAI / mock)
│   │   ├── models.py        shared pydantic schemas
│   │   └── config.py        all tunable settings
│   ├── sample_docs/     seed knowledge base (swap in your own docs)
│   ├── requirements.txt
│   ├── .env.example
│   └── run.sh
└── frontend/            React + Vite live verification UI
    └── src/
        ├── App.jsx
        ├── hooks/useHalluciShieldSocket.js
        └── components/SentenceLine.jsx
```

## Running it

**Backend**
```bash
cd backend
cp .env.example .env      # Windows: copy .env.example .env
# Put your OpenAI key in backend/.env:
# OPENAI_API_KEY=sk-...
./run.sh                  # macOS/Linux
# Windows PowerShell: .\run.ps1
```

The backend always loads `backend/.env`, even when it is started from another
directory. When `OPENAI_API_KEY` is set, generation, embeddings, and claim
verification use OpenAI; otherwise the offline mock providers are selected.

**Frontend**
```bash
cd frontend
npm install
npm run dev                # starts on :5173, proxies /ws and /api to :8000
```

Open `http://localhost:5173`, type a prompt, and watch sentences get
tagged **Supported / Contradicted / Unverifiable** live as the response
generates, each with its retrieved evidence and per-claim latency.

## Adding your own knowledge base

`POST /api/ingest` with `{"doc_id": "...", "source": "...", "text": "..."}`,
or drop `.txt` files into `backend/sample_docs/` (auto-loaded on startup).

## Scaling to production

This reference implementation optimizes for **clarity and zero-dependency
runnability**. To take it further:

- **Vector store**: swap `vector_store.py`'s numpy cosine search for FAISS
  (`IndexHNSWFlat`) or pgvector with an HNSW index — sub-10ms retrieval at
  scale instead of O(n) numpy matmul.
- **Verifier**: replace the single-call LLM entailment with a small,
  distilled NLI cross-encoder (e.g. a DeBERTa-based entailment model)
  served locally for <50ms verification instead of a network round trip.
- **Concurrency**: `MAX_CONCURRENT_VERIFICATIONS` in `config.py` caps
  parallel verification calls per connection — tune based on your
  LLM/embedding provider's rate limits.
- **Caching**: cache embeddings for frequently-asked claims/questions to
  cut repeated retrieval latency.
- **Horizontal scaling**: the WebSocket handler is stateless per-connection
  aside from the shared `vector_store` singleton — run multiple uvicorn
  workers behind a load balancer, with the vector store backed by a shared
  service (pgvector/Pinecone/Weaviate) instead of an in-process index.
