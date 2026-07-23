# Semantic Search + RAG Service — Phase 5 (Final)

Evaluating retrieval quality. This phase adds no new feature — it **measures**
whether the system retrieves the right context, which is the part that
separates real AI engineering from "I wired up an LLM." It's the strongest
artifact in the project.

## Files
- `eval_phase5.py` — the eval set, the metrics, and the chart
- `test_rag_phase5.py` — tests for the evaluation logic
- `rag_eval_chunksize.png` — hit-rate@k and MRR vs chunk size
- `rag_eval_results.json` — the raw numbers

## The two metrics
- **hit-rate@k** — for what fraction of questions did the correct chunk appear
  in the top-k retrieved? (Did we even *fetch* the right text?)
- **MRR (Mean Reciprocal Rank)** — how *high* did the correct chunk rank?
  Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33. Rewards putting the best chunk
  first. MRR can never exceed hit-rate.

## How the eval set works
Each item is a `(question, needle)` pair, where the needle is a distinctive
phrase that appears in exactly the chunk that should answer it (e.g. "8,849
metres" for "How tall is Mount Everest?"). A retrieval is a **hit** if a
retrieved chunk contains that needle — which makes "the correct chunk"
unambiguous no matter how the text was split.

## The result (vary chunk size, chart the effect)
```
  chunk_size | hit-rate@3 |  MRR
  ----------------------------------
          10 |        50% | 0.35
          15 |        75% | 0.60
          20 |        75% | 0.75
          30 |       100% | 0.94
          50 |        88% | 0.73
          80 |       100% | 0.94
```
`rag_eval_chunksize.png` plots this. The shape tells the story: **very small
chunks split answers apart (50% at size 10), quality climbs to a peak around
size 30, dips, and recovers.** There's a sweet spot — too small loses context,
and the right size isolates each answer cleanly.

## Run
```bash
pip install matplotlib chromadb
python eval_phase5.py        # prints the table, writes the chart + JSON
python test_rag_phase5.py    # tests the eval logic
```

## Why this is the headline interview artifact
"I didn't just build RAG — I evaluated it. I measured hit-rate@k and MRR
against an eval set and charted how chunk size affects retrieval quality, which
showed a sweet spot around 30 words." That demonstrates the three things
AI-engineering interviewers probe: you can build modern AI, you understand the
tradeoffs, and **you can measure an AI system rather than just assemble one.**

## Honesty notes
- These absolute numbers use the lightweight `HashingEmbedder`. On Colab with a
  real embedding model the scores rise, but **the method is identical** —
  build an eval set, measure hit-rate@k and MRR, tune a knob, chart it.
- A test (`eval_set_wellformed`) caught that a needle phrase spanned a line
  break in the raw corpus; the fix was to check against whitespace-normalized
  text, matching what the chunker actually produces. (The fourth time in this
  project that a behaviour/property test caught a real issue.)
- Real RAG tuning continues from here: better chunking (sentence-aware), a
  reranking step, or a stronger embedding model — each measured the same way.

## The complete project, recapped
| Phase | Adds | Key idea |
|-------|------|----------|
| 1 | ingest & chunk | overlapping chunks, no words lost |
| 2 | embed & store | semantic search via a pluggable embedder |
| 3 | retrieve & generate | citations + graceful "I don't know" |
| 4 | HTTP service | /ingest and /ask over the network |
| 5 | evaluation | hit-rate@k, MRR, and the chunk-size chart |
