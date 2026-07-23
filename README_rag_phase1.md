# Semantic Search + RAG Service — Phase 1

Document ingest and chunking. No embeddings yet — this phase gets the
**chunking** right in isolation, because retrieval quality later depends
heavily on how documents are split.

## Files
- `ingest_phase1.py` — loaders (text + PDF) and the chunker + a demo
- `test_rag_phase1.py` — tests (no embeddings, no network)

## What it does
Turns a document into overlapping, word-based **chunks**, each carrying
metadata (source file, chunk index, starting word offset):

```python
from ingest_phase1 import ingest, chunk_text
chunks = ingest("mydoc.pdf", chunk_size=200, overlap=40)
# or chunk a string directly:
chunks = chunk_text(text, source="notes", chunk_size=200, overlap=40)
```

- `chunk_size` = words per chunk (the context each chunk carries)
- `overlap` = words shared with the previous chunk (so an answer split across a
  boundary isn't lost). The window advances by `chunk_size - overlap` each step.

Supports `.txt`, `.md`, and `.pdf` (PDF text via pypdf — the same approach as the
Document Authenticity Checker).

## Run
```bash
pip install pypdf
python ingest_phase1.py       # demo: chunks a sample passage
python test_rag_phase1.py     # tests
```

## Why chunking matters (the whole point of Phase 1)
- **Too big** → retrieval returns a blob with the answer buried in noise.
- **Too small** → a chunk loses the context needed to be meaningful.
- **Overlap** → prevents losing an answer that straddles a chunk boundary.

These are tuning knobs you'll *evaluate* in Phase 5, not guess. Getting the
mechanics correct and tested now means later phases build on solid ground.

## Tests cover
- whitespace normalization
- correct chunk boundaries (window starts and step size)
- **overlap actually overlaps** (consecutive chunks share `overlap` words)
- **full coverage** (no source word is ever lost across the chunk set)
- the last chunk reaches the end of the document
- short text → one chunk; empty text → no chunks
- metadata present (source + index)
- bad config rejected (e.g. overlap >= chunk_size)
- ingesting a real text file from disk

> Note: a property test ("no words lost") caught a boundary miscalculation
> during development — verifying the *property* independently of the arithmetic
> is exactly why it surfaced. Good tests check what must be true, not what you
> think the numbers are.

## Still deferred
- Embeddings + vector store + semantic search → **Phase 2**
- The RAG loop (retrieve + generate with citations) → **Phase 3**
- HTTP service (/ingest, /ask) → **Phase 4**
- Retrieval-quality evaluation (hit-rate@k, MRR) + chart → **Phase 5**
