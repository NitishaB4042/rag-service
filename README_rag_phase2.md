# Semantic Search + RAG Service — Phase 2

Embed the Phase 1 chunks, store them in a vector database (Chroma), and search
by **meaning**: embed a question, retrieve the nearest chunks.

## Files
- `store_phase2.py` — pluggable embedder + Chroma vector store + search + demo
- `test_rag_phase2.py` — tests (in-memory store, controllable embedder)

## What it does
```python
from ingest_phase1 import chunk_text
from store_phase2 import HashingEmbedder, VectorStore

store = VectorStore(HashingEmbedder())
store.add(chunk_text(text, source="doc.txt", chunk_size=200, overlap=40))
results = store.search("my question", k=3)   # -> [{text, source, index, score}, ...]
```

## The pluggable embedder (an important design choice)
The store and search code never depend on one embedding provider. An embedder
is anything with `.embed(list[str]) -> list[vector]`. Two ship here:

- **HashingEmbedder** — zero heavy dependencies. Maps words to vector positions
  by hashing and normalizes. It demonstrates and tests the *entire* pipeline
  anywhere (even a tablet sandbox), but it matches on shared **words**, not
  meaning — so retrieval isn't semantically smart.
- **SentenceTransformerEmbedder** — the real one (commented out at the top of
  `store_phase2.py`). On Colab or any real machine, `pip install
  sentence-transformers`, uncomment it, and pass it instead:
  ```python
  store = VectorStore(SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
  ```
  Everything else is identical — that's the whole point of the interface.

> Why this matters: a real embedding model (PyTorch-based) is large and
> needs a proper environment. Decoupling the embedder means the pipeline,
> store, search, tests, and later phases all work regardless of which embedder
> is plugged in. On a tablet you develop with HashingEmbedder; on Colab you
> flip to the real model for genuine meaning-based matching. Same code.

## Run
```bash
pip install chromadb
python store_phase2.py        # demo: store mixed text, run a few searches
python test_rag_phase2.py     # tests
```

## Tests cover
- add + count
- **search returns the nearest chunk by meaning** (using a controllable 3-D
  fake embedder, so the expected nearest neighbour is known exactly)
- scores descend (most similar first)
- `k` limits the number of results
- metadata (source, index) round-trips through the store
- empty add is a no-op
- the HashingEmbedder produces correctly-shaped, L2-normalized vectors

> A test caught a real isolation bug: two `VectorStore` objects in one process
> were sharing a Chroma collection, so leftover chunks contaminated results.
> Fixed by giving each store a unique collection name by default. Tests that
> check behaviour (not just "no error") surface exactly this kind of issue.

## Notes for Colab
```python
!pip install sentence-transformers chromadb
# uncomment SentenceTransformerEmbedder in store_phase2.py, then:
store = VectorStore(SentenceTransformerEmbedder())
```
With the real embedder, "how do I reset my password?" will match "I forgot my
login credentials" even with no shared words — true semantic search.

## Still deferred
- The RAG loop: retrieve + generate an answer with citations → **Phase 3**
- HTTP service (/ingest, /ask) → **Phase 4**
- Retrieval-quality evaluation (hit-rate@k, MRR) + chart → **Phase 5**
