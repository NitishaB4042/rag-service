"""
Tests for RAG Phase 2 — embed, store, search.
Uses an in-memory Chroma store and a controllable embedder, so results are
deterministic and don't depend on a heavy ML model.
Run: python test_rag_phase2.py
"""
from ingest_phase1 import Chunk
import store_phase2 as m


# A tiny controllable embedder: maps known texts to known vectors, so we can
# assert exact nearest-neighbour behaviour independent of any ML model.
class FakeEmbedder(m.Embedder):
    name = "fake-3d"
    dim = 3
    TABLE = {
        "dog":   [1.0, 0.0, 0.0],
        "puppy": [0.9, 0.1, 0.0],   # close to dog
        "tax":   [0.0, 1.0, 0.0],   # far from dog
        "audit": [0.0, 0.9, 0.1],   # close to tax
        "plant": [0.0, 0.0, 1.0],
    }
    def embed(self, texts):
        out = []
        for t in texts:
            key = t.strip().lower().split()[0] if t.strip() else ""
            out.append(self.TABLE.get(key, [0.0, 0.0, 0.0]))
        return out


def _chunks(words):
    return [Chunk(text=w, source="t.txt", index=i, start_word=i)
            for i, w in enumerate(words)]


def test_add_and_count():
    store = m.VectorStore(FakeEmbedder())
    n = store.add(_chunks(["dog", "tax", "plant"]))
    assert n == 3
    assert store.count() == 3
    print("  add_and_count: PASS")


def test_search_finds_nearest_by_meaning():
    store = m.VectorStore(FakeEmbedder())
    store.add(_chunks(["dog", "tax", "plant", "audit"]))
    # "puppy" is nearest to "dog" in our fake space
    top = store.search("puppy", k=1)
    assert top[0]["text"] == "dog", top[0]["text"]
    # "audit" query should surface the tax/audit cluster, not dog/plant
    top2 = store.search("audit", k=2)
    texts = {r["text"] for r in top2}
    assert "tax" in texts or "audit" in texts, texts
    assert "dog" not in texts, texts
    print("  search_finds_nearest_by_meaning: PASS")


def test_scores_descend():
    store = m.VectorStore(FakeEmbedder())
    store.add(_chunks(["dog", "tax", "plant", "audit"]))
    results = store.search("puppy", k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), scores  # most similar first
    print("  scores_descend (most similar first): PASS")


def test_k_limits_results():
    store = m.VectorStore(FakeEmbedder())
    store.add(_chunks(["dog", "tax", "plant", "audit", "puppy"]))
    assert len(store.search("dog", k=2)) == 2
    assert len(store.search("dog", k=5)) == 5
    print("  k_limits_results: PASS")


def test_metadata_round_trips():
    store = m.VectorStore(FakeEmbedder())
    store.add([Chunk(text="dog", source="animals.txt", index=7, start_word=42)])
    r = store.search("puppy", k=1)[0]
    assert r["source"] == "animals.txt" and r["index"] == 7, r
    print("  metadata_round_trips: PASS")


def test_empty_add():
    store = m.VectorStore(FakeEmbedder())
    assert store.add([]) == 0
    assert store.count() == 0
    print("  empty_add: PASS")


def test_hashing_embedder_shape_and_norm():
    emb = m.HashingEmbedder(dim=64)
    vecs = emb.embed(["hello world", "hello"])
    assert len(vecs) == 2 and len(vecs[0]) == 64
    import math
    norm = math.sqrt(sum(v*v for v in vecs[0]))
    assert abs(norm - 1.0) < 1e-6, norm     # L2-normalized
    print("  hashing_embedder_shape_and_norm: PASS")


if __name__ == "__main__":
    print("Running RAG Phase 2 tests:")
    test_add_and_count()
    test_search_finds_nearest_by_meaning()
    test_scores_descend()
    test_k_limits_results()
    test_metadata_round_trips()
    test_empty_add()
    test_hashing_embedder_shape_and_norm()
    print("All tests passed.")
