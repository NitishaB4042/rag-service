# Semantic Search + RAG Service — Phase 4

The RAG pipeline wrapped as a real HTTP service (FastAPI). Ingest documents,
then ask questions and get cited, grounded answers over HTTP. Reuses the
rate-limiter's service skeleton almost wholesale.

## Files
- `rag_service.py` — the FastAPI app
- `test_rag_phase4.py` — tests (drive the real app in-process via httpx ASGI)

## Endpoints
- `GET  /health` → `{status, chunks}`
- `POST /ingest` → `{documents:[{text, source?}], chunk_size?, overlap?}` →
  `{ingested_chunks, total_chunks}`
- `POST /ask` → `{question, k?}` → `{answer, grounded, sources:[...]}`
- `POST /reset` → clears the in-memory store (for testing/demos)

## Run
```bash
pip install fastapi "uvicorn[standard]" httpx chromadb
uvicorn rag_service:app --port 8000
```
```bash
# ingest a document
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"text":"The capital of France is Paris.","source":"geo"}]}'

# ask a question -> grounded, cited answer
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is the capital of France?"}'
# {"answer":"The capital of France is Paris.","grounded":true,
#  "sources":[{"source":"geo","index":0,"score":0.41}]}

# ask something not in the docs -> truthful refusal
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"boiling point of mercury?"}'
# {"answer":"I don't know based on the provided documents.","grounded":false,"sources":[]}
```

## Behaviour worth noting
- **/ask before any /ingest → 409 Conflict** (nothing to answer from), rather
  than a confusing empty answer.
- **Validation** via Pydantic: empty document list or `overlap >= chunk_size`
  → 422 before our logic runs.
- **Grounded answers carry sources; refusals carry none** — the trust contract
  from Phase 3, now over HTTP.

## Test
```bash
python test_rag_phase4.py
```
Tests drive the real app in-process (no separate server) and cover: the full
ingest→ask flow with citations, ask-before-ingest 409, the refusal path,
input validation, and that reset clears the store.

## Going real on Colab
In `build_pipeline()` (and the embedder/LLM modules), swap
`HashingEmbedder → SentenceTransformerEmbedder` and
`ExtractiveLLM → OpenAILLM`. Nothing else in the service changes — the API,
validation, and tests are all embedder/LLM-agnostic. That decoupling is the
payoff of the pluggable interfaces from Phases 2 and 3.

## Design notes (interview)
- **State built once at startup** via FastAPI `lifespan`, held in memory across
  requests (not rebuilt per call).
- **Stateless API shape**: in production you'd point the store at a shared,
  persistent vector DB so many service copies share one index — the same
  "stateless servers + shared store" pattern as the crawler and rate limiter.
- **A fixed collection name reintroduced shared state across resets** during
  development; using a fresh unique collection per build fixed it. (Second time
  in this project that behaviour-checking tests caught a state-isolation bug.)

## Still deferred
- Retrieval-quality evaluation (hit-rate@k, MRR) + chart → **Phase 5** (finale)
