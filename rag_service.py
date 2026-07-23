"""
Semantic Search + RAG Service — Phase 4: HTTP API (FastAPI).

Turns the RAG pipeline into a real service other systems call over HTTP.

Endpoints:
  GET  /health                     -> {"status": "ok", "chunks": N}
  POST /ingest  {documents:[{text, source?}], chunk_size?, overlap?}
                                    -> {ingested_chunks, total_chunks}
  POST /ask     {question, k?}     -> {answer, grounded, sources:[...]}
  POST /reset                      -> clears the in-memory store (for testing)

Run the server:
    uvicorn rag_service:app --port 8000
  (or: python rag_service.py)

Then:
    curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
         -d '{"documents":[{"text":"The capital of France is Paris.","source":"geo"}]}'
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"What is the capital of France?"}'
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingest_phase1 import chunk_text
from store_phase2 import HashingEmbedder, VectorStore
from rag_phase3 import RAG, ExtractiveLLM


# ---- shared state, built once at startup ----
_state: dict = {}


def build_pipeline():
    """Construct the store + RAG pipeline.

    Swap HashingEmbedder -> SentenceTransformerEmbedder and ExtractiveLLM ->
    OpenAILLM here (and in their modules) to run the real models on Colab.
    Nothing else in the service changes.
    """
    embedder = HashingEmbedder()
    store = VectorStore(embedder)   # unique collection per build -> reset truly clears
    llm = ExtractiveLLM()
    rag = RAG(store, llm, k=3)
    return store, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    store, rag = build_pipeline()
    _state["store"] = store
    _state["rag"] = rag
    yield
    _state.clear()


app = FastAPI(title="Semantic Search + RAG Service", lifespan=lifespan)


# ---- request/response shapes ----
class DocIn(BaseModel):
    text: str
    source: str = "inline"


class IngestRequest(BaseModel):
    documents: list[DocIn] = Field(min_length=1)
    chunk_size: int = Field(default=200, gt=0)
    overlap: int = Field(default=40, ge=0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, gt=0, le=20)


@app.get("/health")
async def health():
    store = _state.get("store")
    if store is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok", "chunks": store.count()}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    if req.overlap >= req.chunk_size:
        return JSONResponse(status_code=422,
                            content={"error": "overlap must be < chunk_size"})
    store = _state["store"]
    added = 0
    for doc in req.documents:
        chunks = chunk_text(doc.text, source=doc.source,
                            chunk_size=req.chunk_size, overlap=req.overlap)
        added += store.add(chunks)
    return {"ingested_chunks": added, "total_chunks": store.count()}


@app.post("/ask")
async def ask(req: AskRequest):
    store = _state["store"]
    if store.count() == 0:
        return JSONResponse(status_code=409,
                            content={"error": "no documents ingested yet"})
    rag = _state["rag"]
    ans = rag.ask(req.question, k=req.k)
    return {
        "answer": ans.text,
        "grounded": ans.grounded,
        "sources": ans.sources,
    }


@app.post("/reset")
async def reset():
    store, rag = build_pipeline()
    _state["store"] = store
    _state["rag"] = rag
    return {"reset": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
