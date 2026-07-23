"""
Semantic Search + RAG Service — Phase 3: the RAG loop.

Ties everything together. On a question:
  1. retrieve the top-k nearest chunks (Phase 2),
  2. build a prompt that says "answer using ONLY this context",
  3. call a language model,
  4. return the answer WITH CITATIONS (which chunks/sources it used),
     and a truthful "I don't know" when the context doesn't support an answer.

Design choice: the LLM is PLUGGABLE behind a tiny interface, just like the
embedder in Phase 2. Two ship here:

  - ExtractiveLLM : dependency-free, deterministic. Doesn't "write" prose; it
                    grounds strictly in the context and is perfect for testing
                    the RAG plumbing (citations, the I-don't-know path) anywhere.
  - OpenAILLM     : the real generator for Colab / a real machine (commented
                    out). Same interface, so nothing else changes.

Run the demo:   python rag_phase3.py
Run the tests:  python test_phase3.py
"""

import re
from dataclasses import dataclass

from ingest_phase1 import chunk_text
from store_phase2 import HashingEmbedder, VectorStore


# ===========================================================================
# LLM interface — anything with .name and .generate(prompt) -> str
# ===========================================================================
class LLM:
    name = "base"
    def generate(self, prompt: str, context_blocks: list[str], question: str) -> str:
        raise NotImplementedError


# A phrase the prompt asks the model to emit when the context is insufficient.
IDK = "I don't know based on the provided documents."


class ExtractiveLLM(LLM):
    """A dependency-free stand-in that grounds STRICTLY in the context.

    It does not generate free prose. Instead it returns the sentence(s) from the
    retrieved context that best overlap the question's words. If nothing
    overlaps, it returns the I-don't-know phrase. This lets us test the RAG
    plumbing \u2014 grounding, citations, the refusal path \u2014 deterministically,
    with no API and no heavy model. Swap in OpenAILLM for real prose answers.
    """
    name = "extractive-stub"
    MIN_OVERLAP = 1            # need at least this many shared content words

    def _content_words(self, text: str) -> set[str]:
        stop = {"the","a","an","of","to","in","is","are","and","or","do","i",
                "how","what","why","with","for","on","does","my","me","it",
                "this","that","using","use","can","you","your"}
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in stop and len(w) > 2}

    def generate(self, prompt, context_blocks, question):
        q_words = self._content_words(question)
        best_sentence, best_overlap = None, 0
        for block in context_blocks:
            for sentence in re.split(r"(?<=[.!?])\s+", block):
                overlap = len(q_words & self._content_words(sentence))
                if overlap > best_overlap:
                    best_overlap, best_sentence = overlap, sentence.strip()
        if best_sentence is None or best_overlap < self.MIN_OVERLAP:
            return IDK
        return best_sentence


# Real generator for Colab / a real machine. Uncomment to use:
#
# from openai import OpenAI
# class OpenAILLM(LLM):
#     name = "gpt-4o-mini"
#     def __init__(self, model="gpt-4o-mini"):
#         self.client = OpenAI()           # reads OPENAI_API_KEY from env
#         self.model = model
#     def generate(self, prompt, context_blocks, question):
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0)
#         return resp.choices[0].message.content.strip()


# ===========================================================================
# Prompt construction \u2014 the instructions that enforce grounding
# ===========================================================================
def build_prompt(context_blocks: list[str], question: str) -> str:
    """Assemble the grounded prompt. Numbered context lets the model cite [1],[2]."""
    numbered = "\n\n".join(f"[{i+1}] {block}"
                           for i, block in enumerate(context_blocks))
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, reply exactly: "
        f"\"{IDK}\"\n"
        "Cite the context blocks you used, like [1] or [2].\n\n"
        f"Context:\n{numbered}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


@dataclass
class Answer:
    text: str
    sources: list[dict]      # [{source, index, score}] for the chunks used
    grounded: bool           # False when the model said "I don't know"


# ===========================================================================
# The RAG pipeline
# ===========================================================================
class RAG:
    def __init__(self, store: VectorStore, llm: LLM, k: int = 3):
        self.store = store
        self.llm = llm
        self.k = k

    def ask(self, question: str, k: int | None = None) -> Answer:
        k = k or self.k
        hits = self.store.search(question, k=k)
        if not hits:
            return Answer(text=IDK, sources=[], grounded=False)

        context_blocks = [h["text"] for h in hits]
        prompt = build_prompt(context_blocks, question)
        text = self.llm.generate(prompt, context_blocks, question)

        grounded = text.strip() != IDK
        # only report sources when we actually produced a grounded answer
        sources = []
        if grounded:
            for h in hits:
                sources.append({"source": h["source"], "index": h["index"],
                                "score": round(h["score"], 3)})
        return Answer(text=text, sources=sources, grounded=grounded)


# ===========================================================================
# Demo
# ===========================================================================
SAMPLE = """
Retrieval-augmented generation answers questions using a language model
together with your own documents. Documents are first split into chunks. Each
chunk is turned into an embedding and stored in a vector database. At question
time, the question is embedded and the nearest chunks are retrieved, then passed
to the model as context. The capital of France is Paris. Photosynthesis lets
plants convert sunlight into chemical energy stored as sugars.
""".strip()


def _demo():
    store = VectorStore(HashingEmbedder())
    store.add(chunk_text(SAMPLE, source="notes.txt", chunk_size=20, overlap=5))
    rag = RAG(store, ExtractiveLLM(), k=3)

    questions = [
        "How are documents turned into something searchable?",
        "What is the capital of France?",
        "What is the boiling point of mercury?",   # not in the docs -> I don't know
    ]
    for q in questions:
        ans = rag.ask(q)
        print(f"Q: {q}")
        print(f"A: {ans.text}")
        if ans.grounded:
            cites = ", ".join(f"{s['source']}#{s['index']}" for s in ans.sources)
            print(f"   sources: {cites}")
        else:
            print("   (no grounded answer \u2014 refused rather than guess)")
        print()

    print("Note: this demo uses the dependency-free ExtractiveLLM, which quotes "
          "the best-matching sentence from the context. On Colab, switch to "
          "OpenAILLM for fluent generated prose. The grounding, citations, and "
          "refusal logic are identical either way.")


if __name__ == "__main__":
    _demo()
