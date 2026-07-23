"""
Semantic Search + RAG Service — Phase 5: evaluate retrieval quality.

Anyone can wire up RAG; the engineer's job is to MEASURE whether it retrieves
the right context. This phase builds a small eval set (question -> the source
chunk that truly answers it), then measures:

  - hit-rate@k : for what fraction of questions did the correct chunk appear
                 in the top-k retrieved? (Did we even fetch the right text?)
  - MRR        : Mean Reciprocal Rank \u2014 how HIGH did the correct chunk rank?
                 rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33 ... rewards
                 putting the best chunk first.

Then it varies a design knob (chunk size) and charts how hit-rate@k responds.

Run:
    python eval_phase5.py

Outputs:
    rag_eval_chunksize.png      \u2014 hit-rate@3 vs chunk size
    rag_eval_results.json
"""

import json
from dataclasses import dataclass

from ingest_phase1 import chunk_text
from store_phase2 import HashingEmbedder, VectorStore
# On Colab, swap HashingEmbedder -> SentenceTransformerEmbedder for real numbers.


# ===========================================================================
# A small corpus and an eval set.
# Each eval item: a question + a "needle" \u2014 a distinctive phrase that appears
# in exactly the chunk that should answer it. We mark a retrieval a HIT if a
# retrieved chunk contains that needle. (Using a needle phrase makes "the
# correct chunk" unambiguous regardless of how the text was chunked.)
# ===========================================================================
CORPUS = """
The Eiffel Tower is located in Paris and was completed in 1889 for the World's
Fair. The Great Wall of China stretches thousands of kilometres and was built
over many centuries to defend against invasions. Mount Everest is the highest
mountain above sea level, standing about 8,849 metres tall in the Himalayas.
The Amazon rainforest produces a large share of the world's oxygen and hosts
enormous biodiversity. The human heart beats roughly one hundred thousand times
per day, pumping blood through the circulatory system. Photosynthesis allows
green plants to convert sunlight, water, and carbon dioxide into glucose and
oxygen. The speed of light in a vacuum is about three hundred thousand
kilometres per second. Honey never spoils; archaeologists have found edible
honey in ancient Egyptian tombs thousands of years old. The Pacific Ocean is
the largest and deepest of Earth's oceans. Water boils at one hundred degrees
Celsius at standard atmospheric pressure.
""".strip()

# question -> needle phrase that identifies the correct chunk
EVAL = [
    ("When was the Eiffel Tower completed?",          "1889"),
    ("How tall is Mount Everest?",                     "8,849 metres"),
    ("What does photosynthesis produce?",              "glucose and oxygen"),
    ("How fast does light travel?",                    "three hundred thousand"),
    ("Does honey go bad?",                             "Honey never spoils"),
    ("Which is the largest ocean?",                    "Pacific Ocean"),
    ("At what temperature does water boil?",           "one hundred degrees Celsius"),
    ("How often does the human heart beat?",           "one hundred thousand times"),
]


@dataclass
class EvalResult:
    chunk_size: int
    k: int
    hit_rate: float
    mrr: float


def evaluate(chunk_size: int, overlap: int, k: int) -> EvalResult:
    """Build a store at a given chunk size, then score the eval set."""
    store = VectorStore(HashingEmbedder())
    store.add(chunk_text(CORPUS, source="facts.txt",
                        chunk_size=chunk_size, overlap=overlap))

    hits = 0
    reciprocal_ranks = 0.0
    for question, needle in EVAL:
        results = store.search(question, k=k)
        # find the rank (1-based) of the first retrieved chunk containing the needle
        rank = None
        for i, r in enumerate(results, start=1):
            if needle.lower() in r["text"].lower():
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank
    n = len(EVAL)
    return EvalResult(chunk_size=chunk_size, k=k,
                      hit_rate=hits / n, mrr=reciprocal_ranks / n)


def main():
    K = 3
    overlap = 5
    chunk_sizes = [10, 15, 20, 30, 50, 80]

    print(f"Evaluating retrieval over {len(EVAL)} questions, k={K}\n")
    print(f"  {'chunk_size':>10} | {'hit-rate@'+str(K):>11} | {'MRR':>6}")
    print("  " + "-" * 34)
    results = []
    for cs in chunk_sizes:
        res = evaluate(cs, overlap, K)
        results.append(res)
        print(f"  {cs:>10} | {res.hit_rate*100:>10.0f}% | {res.mrr:>6.2f}")

    best = max(results, key=lambda r: (r.hit_rate, r.mrr))
    print(f"\n  Best chunk size for this corpus: {best.chunk_size} "
          f"(hit-rate {best.hit_rate*100:.0f}%, MRR {best.mrr:.2f})")

    with open("rag_eval_results.json", "w") as f:
        json.dump([{"chunk_size": r.chunk_size, "k": r.k,
                    "hit_rate": round(r.hit_rate, 3), "mrr": round(r.mrr, 3)}
                   for r in results], f, indent=2)
    print("  wrote rag_eval_results.json")

    # chart: hit-rate@k vs chunk size
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r.chunk_size for r in results]
        hr = [r.hit_rate * 100 for r in results]
        mrr = [r.mrr * 100 for r in results]
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.plot(xs, hr, "o-", color="#2E75B6", linewidth=2.2, markersize=7,
                label=f"hit-rate@{K}")
        ax.plot(xs, mrr, "s--", color="#B9770E", linewidth=1.8, markersize=6,
                label="MRR (x100)")
        ax.set_xlabel("chunk size (words)")
        ax.set_ylabel("retrieval quality (%)")
        ax.set_title("Retrieval quality vs chunk size\n(too small loses context; quality peaks at a mid-range chunk size)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("rag_eval_chunksize.png", dpi=130)
        print("  wrote rag_eval_chunksize.png")
    except ImportError:
        print("  (matplotlib not installed \u2014 skipped chart)")

    print("\nNote: these numbers use the lightweight HashingEmbedder. On Colab "
          "with a real embedding model the absolute scores rise, but the METHOD "
          "\u2014 measure hit-rate@k and MRR, then tune a knob \u2014 is exactly the same.")


if __name__ == "__main__":
    main()
