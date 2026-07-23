"""
Tests for RAG Phase 3 — the retrieve+generate loop.
Uses controllable embedder + LLM so behaviour is deterministic and needs no
heavy model or API.
Run: python test_rag_phase3.py
"""
from ingest_phase1 import Chunk
import store_phase2 as s
import rag_phase3 as m


class FakeEmbedder(s.Embedder):
    """3-D space: query word maps near the matching chunk word."""
    name = "fake-3d"; dim = 3
    TABLE = {"dog":[1.0,0,0], "puppy":[0.95,0.05,0],
             "tax":[0,1.0,0], "audit":[0,0.95,0.05],
             "plant":[0,0,1.0]}
    def embed(self, texts):
        return [self.TABLE.get(t.strip().lower().split()[0] if t.strip() else "",
                               [0.0,0.0,0.0]) for t in texts]


def _store(words):
    st = s.VectorStore(FakeEmbedder())
    st.add([Chunk(text=w, source="t.txt", index=i, start_word=i)
            for i, w in enumerate(words)])
    return st


def test_build_prompt_numbers_and_idk():
    p = m.build_prompt(["alpha", "beta"], "what?")
    assert "[1] alpha" in p and "[2] beta" in p
    assert m.IDK in p                       # the refusal instruction is present
    assert "Question: what?" in p
    print("  build_prompt_numbers_and_idk: PASS")


def test_grounded_answer_has_sources():
    rag = m.RAG(_store(["dog", "tax", "plant"]), m.ExtractiveLLM(), k=3)
    ans = rag.ask("dog")                    # extractive LLM will quote "dog"
    assert ans.grounded is True
    assert ans.text == "dog", ans.text
    assert len(ans.sources) > 0
    assert all("source" in src and "index" in src for src in ans.sources)
    print("  grounded_answer_has_sources: PASS")


def test_refuses_when_context_irrelevant():
    # ExtractiveLLM returns IDK when no question word overlaps any context word
    rag = m.RAG(_store(["dog", "tax", "plant"]), m.ExtractiveLLM(), k=3)
    ans = rag.ask("mercury")                # no overlap with stored words
    assert ans.grounded is False
    assert ans.text == m.IDK
    assert ans.sources == []                # no citations on a refusal
    print("  refuses_when_context_irrelevant: PASS")


def test_empty_store_refuses():
    rag = m.RAG(s.VectorStore(FakeEmbedder()), m.ExtractiveLLM(), k=3)
    ans = rag.ask("dog")
    assert ans.grounded is False and ans.sources == []
    print("  empty_store_refuses: PASS")


def test_llm_is_pluggable():
    # a custom LLM that always answers a fixed string -> RAG should use it
    class AlwaysLLM(m.LLM):
        name = "always"
        def generate(self, prompt, blocks, question):
            return "fixed answer [1]"
    rag = m.RAG(_store(["dog", "tax"]), AlwaysLLM(), k=2)
    ans = rag.ask("dog")
    assert ans.text == "fixed answer [1]"
    assert ans.grounded is True and len(ans.sources) > 0
    print("  llm_is_pluggable: PASS")


def test_k_controls_context_size():
    captured = {}
    class CapturingLLM(m.LLM):
        name = "cap"
        def generate(self, prompt, blocks, question):
            captured["n"] = len(blocks)
            return "ok"
    rag = m.RAG(_store(["dog","tax","plant"]), CapturingLLM(), k=3)
    rag.ask("dog", k=2)
    assert captured["n"] == 2, captured      # k=2 -> 2 context blocks
    print("  k_controls_context_size: PASS")


if __name__ == "__main__":
    print("Running RAG Phase 3 tests:")
    test_build_prompt_numbers_and_idk()
    test_grounded_answer_has_sources()
    test_refuses_when_context_irrelevant()
    test_empty_store_refuses()
    test_llm_is_pluggable()
    test_k_controls_context_size()
    print("All tests passed.")
