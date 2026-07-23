"""
Tests for RAG Phase 5 — the evaluation logic. No heavy model needed.
Run: python test_rag_phase5.py
"""
import eval_phase5 as m


def test_metrics_in_valid_range():
    res = m.evaluate(chunk_size=30, overlap=5, k=3)
    assert 0.0 <= res.hit_rate <= 1.0, res.hit_rate
    assert 0.0 <= res.mrr <= 1.0, res.mrr
    # MRR can never exceed hit-rate (each hit contributes <= 1.0 to MRR)
    assert res.mrr <= res.hit_rate + 1e-9, (res.mrr, res.hit_rate)
    print("  metrics_in_valid_range: PASS")


def test_reasonable_chunk_size_retrieves_well():
    # at a sensible chunk size, retrieval should find most answers
    res = m.evaluate(chunk_size=30, overlap=5, k=3)
    assert res.hit_rate >= 0.7, res.hit_rate
    print(f"  reasonable_chunk_size_retrieves_well (hit-rate={res.hit_rate:.2f}): PASS")


def test_larger_k_never_lowers_hit_rate():
    # retrieving more candidates can only help (or tie) hit-rate
    small_k = m.evaluate(chunk_size=20, overlap=5, k=1)
    big_k = m.evaluate(chunk_size=20, overlap=5, k=5)
    assert big_k.hit_rate >= small_k.hit_rate - 1e-9, (small_k.hit_rate, big_k.hit_rate)
    print("  larger_k_never_lowers_hit_rate: PASS")


def test_mrr_rewards_top_rank():
    # k=1 MRR equals hit-rate (only rank-1 hits possible)
    res = m.evaluate(chunk_size=30, overlap=5, k=1)
    assert abs(res.mrr - res.hit_rate) < 1e-9, (res.mrr, res.hit_rate)
    print("  mrr_rewards_top_rank (k=1 -> MRR==hit-rate): PASS")


def test_eval_set_wellformed():
    # every needle must actually exist in the corpus AS THE CHUNKER SEES IT
    # (whitespace-normalized), or the eval would be unfair. The raw corpus has
    # line breaks, so we normalize before checking \u2014 the same normalization the
    # chunker applies.
    import ingest_phase1 as ing
    corpus_norm = ing.normalize_whitespace(m.CORPUS).lower()
    for q, needle in m.EVAL:
        assert needle.lower() in corpus_norm, f"needle not in corpus: {needle}"
    print("  eval_set_wellformed (all needles present): PASS")


if __name__ == "__main__":
    print("Running RAG Phase 5 tests:")
    test_metrics_in_valid_range()
    test_reasonable_chunk_size_retrieves_well()
    test_larger_k_never_lowers_hit_rate()
    test_mrr_rewards_top_rank()
    test_eval_set_wellformed()
    print("All tests passed.")
