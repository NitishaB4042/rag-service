"""
Tests for RAG Phase 1 — ingest & chunking. No embeddings, no network.
Run: python test_rag_phase1.py
"""
import os
import tempfile
import ingest_phase1 as m


def test_normalize_whitespace():
    assert m.normalize_whitespace("a\n\n  b\t c ") == "a b c"
    print("  normalize_whitespace: PASS")


def test_basic_chunking_counts():
    # 100 words, chunk_size 40, overlap 10 -> step 30.
    # starts 0,30,60; the window at 60 covers words 60..99 (reaches the end),
    # so it's 3 chunks, not 4.
    text = " ".join(str(i) for i in range(100))
    chunks = m.chunk_text(text, chunk_size=40, overlap=10)
    starts = [c.start_word for c in chunks]
    assert starts == [0, 30, 60], starts
    assert len(chunks) == 3, len(chunks)
    print("  basic_chunking_counts (starts 0,30,60): PASS")


def test_overlap_actually_overlaps():
    text = " ".join(str(i) for i in range(100))
    chunks = m.chunk_text(text, chunk_size=40, overlap=10)
    # last 10 words of chunk 0 should equal first 10 words of chunk 1
    w0 = chunks[0].text.split()
    w1 = chunks[1].text.split()
    assert w0[-10:] == w1[:10], (w0[-10:], w1[:10])
    print("  overlap_actually_overlaps: PASS")


def test_full_coverage_no_words_lost():
    # every source word must appear in at least one chunk
    text = " ".join(str(i) for i in range(95))
    chunks = m.chunk_text(text, chunk_size=40, overlap=10)
    covered = set()
    for c in chunks:
        for w in c.text.split():
            covered.add(w)
    assert covered == {str(i) for i in range(95)}, "some words were lost"
    print("  full_coverage_no_words_lost: PASS")


def test_last_chunk_reaches_end():
    text = " ".join(str(i) for i in range(95))
    chunks = m.chunk_text(text, chunk_size=40, overlap=10)
    assert chunks[-1].text.split()[-1] == "94", chunks[-1].text.split()[-1]
    print("  last_chunk_reaches_end: PASS")


def test_short_text_one_chunk():
    chunks = m.chunk_text("just a few words", chunk_size=40, overlap=10)
    assert len(chunks) == 1, len(chunks)
    assert chunks[0].text == "just a few words"
    print("  short_text_one_chunk: PASS")


def test_empty_text():
    assert m.chunk_text("   \n  ", chunk_size=40, overlap=10) == []
    print("  empty_text: PASS")


def test_metadata_present():
    chunks = m.chunk_text("a b c d e f", source="doc.txt", chunk_size=3, overlap=1)
    assert all(c.source == "doc.txt" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    print("  metadata_present (source + index): PASS")


def test_bad_config_rejected():
    for cs, ov in [(0, 0), (40, 40), (40, 50), (40, -1)]:
        try:
            m.chunk_text("a b c", chunk_size=cs, overlap=ov)
            assert False, f"should have raised for chunk_size={cs}, overlap={ov}"
        except ValueError:
            pass
    print("  bad_config_rejected: PASS")


def test_ingest_text_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("one two three four five six seven eight")
        path = f.name
    try:
        chunks = m.ingest(path, chunk_size=4, overlap=1)
        assert chunks[0].source == os.path.basename(path)
        assert chunks[0].text.split()[:4] == ["one", "two", "three", "four"]
    finally:
        os.unlink(path)
    print("  ingest_text_file: PASS")


if __name__ == "__main__":
    print("Running RAG Phase 1 tests:")
    test_normalize_whitespace()
    test_basic_chunking_counts()
    test_overlap_actually_overlaps()
    test_full_coverage_no_words_lost()
    test_last_chunk_reaches_end()
    test_short_text_one_chunk()
    test_empty_text()
    test_metadata_present()
    test_bad_config_rejected()
    test_ingest_text_file()
    print("All tests passed.")
