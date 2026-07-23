# Semantic Search + RAG Service — Phase 3

The RAG loop: retrieve relevant chunks, build a grounded prompt, call a
language model, and return an answer **with citations** — plus a truthful
"I don't know" when the documents don't support an answer.

## Files
- `rag_phase3.py` — pluggable LLM + prompt builder + the RAG pipeline + demo
- `test_rag_phase3.py` — tests (controllable embedder + LLM, no API)

## What it does
```python
from store_phase2 import HashingEmbedder, VectorStore
from rag_phase3 import RAG, ExtractiveLLM
from ingest_phase1 import chunk_text

store = VectorStore(HashingEmbedder())
store.add(chunk_text(text, source="doc.txt"))
rag = RAG(store, ExtractiveLLM(), k=3)

ans = rag.ask("What is the capital of France?")
print(ans.text)       # the answer
print(ans.sources)    # [{source, index, score}, ...] — the chunks it used
print(ans.grounded)   # False if it refused (said "I don't know")
```

## The two behaviours that make RAG trustworthy
1. **Citations** — every grounded answer reports which chunks (source + index)
   it drew from, so any claim is verifiable. "My system shows its work."
2. **Graceful refusal** — the prompt instructs the model to answer *only* from
   the retrieved context and to say exactly "I don't know based on the provided
   documents" otherwise. When it refuses, no sources are attached. This is how
   we fence in **hallucination** (confident, made-up answers).

## The pluggable LLM (same idea as Phase 2's embedder)
An LLM is anything with `.generate(prompt, context_blocks, question) -> str`.
Two ship here:

- **ExtractiveLLM** — dependency-free, deterministic. It doesn't write prose;
  it returns the context sentence that best overlaps the question, or the
  I-don't-know phrase if nothing overlaps. Perfect for testing the RAG plumbing
  (grounding, citations, refusal) anywhere, with no API.
- **OpenAILLM** — the real generator (commented out at the top of
  `rag_phase3.py`). On Colab: `pip install openai`, set `OPENAI_API_KEY`,
  uncomment it, and pass it instead. Produces fluent prose. The grounding,
  citation, and refusal logic are **identical** either way.

```python
# on Colab:
rag = RAG(store, OpenAILLM("gpt-4o-mini"), k=4)
```

## Run
```bash
python rag_phase3.py        # demo: grounded answers + a refusal
python test_rag_phase3.py   # tests
```

Demo output shows two grounded, cited answers and one truthful refusal for a
question the documents don't cover (boiling point of mercury).

## Tests cover
- the prompt is numbered ([1], [2]) and contains the refusal instruction
- a grounded answer carries sources
- **it refuses when the context is irrelevant** (no hallucination) — and
  attaches no citations on a refusal
- an empty store refuses
- the LLM is genuinely pluggable (a custom LLM is used by the pipeline)
- `k` controls how many context blocks are passed to the LLM

## Why a stub LLM instead of a real one here?
A real LLM call needs an API key (cost) or a large local model (heavy). Making
the LLM pluggable means the *logic* — retrieval, grounding, citations, the
refusal path — is fully testable and deterministic without either. You develop
and test the pipeline with `ExtractiveLLM`, then flip to `OpenAILLM` on Colab
for real prose. This is the same decoupling used for the embedder in Phase 2.

## Still deferred
- HTTP service (/ingest, /ask with streaming) → **Phase 4**
- Retrieval-quality evaluation (hit-rate@k, MRR) + chart → **Phase 5**
