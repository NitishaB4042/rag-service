"""
Semantic Search + RAG Service — Phase 1: ingest and chunk.

Goal of this phase: turn a document into clean, overlapping CHUNKS with
metadata (source file, position), with no embeddings yet. Chunking quality
makes or breaks retrieval later, so we get it right in isolation first.

Supports plain text and PDF.

Run the demo:   python ingest_phase1.py
Run the tests:  python test_phase1.py
"""

import re
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    """One piece of a document, ready to be embedded later."""
    text: str
    source: str           # which file it came from
    index: int            # position of this chunk within the document (0,1,2,...)
    start_word: int       # word offset where this chunk begins (for traceability)


# ---------------------------------------------------------------------------
# Text cleaning + word splitting
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines into single spaces, trim ends."""
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str) -> list[str]:
    """Split on whitespace into words. Simple and language-agnostic enough here."""
    return text.split()


# ---------------------------------------------------------------------------
# The chunker
# ---------------------------------------------------------------------------
def chunk_text(text: str, source: str = "inline",
               chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split `text` into overlapping word-based chunks.

    chunk_size = words per chunk (the body of context each chunk carries)
    overlap    = words shared with the previous chunk (so answers split across
                 a boundary aren't lost)

    Returns a list of Chunk objects with position metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        # overlap must leave forward progress, or we'd loop forever
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = split_words(normalize_whitespace(text))
    if not words:
        return []

    step = chunk_size - overlap          # how far the window advances each time
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        chunk_str = " ".join(window)
        chunks.append(Chunk(text=chunk_str, source=source,
                            index=idx, start_word=start))
        idx += 1
        if start + chunk_size >= len(words):
            break                        # this window reached the end
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Loaders: text and PDF -> raw text
# ---------------------------------------------------------------------------
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    """Extract text from a PDF. (Reuses the PDF-reading approach from the
    Document Authenticity Checker.)"""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(path: str) -> str:
    """Pick a loader by file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    if ext in (".txt", ".md", ""):
        return load_text_file(path)
    raise ValueError(f"unsupported file type: {ext}")


def ingest(path: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Load a document from disk and chunk it."""
    raw = load_document(path)
    return chunk_text(raw, source=os.path.basename(path),
                      chunk_size=chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
SAMPLE = """
Retrieval-augmented generation, or RAG, is a technique for answering questions
using a language model together with your own documents. Instead of relying on
what the model memorized during training, the system first retrieves relevant
passages from a document collection and then asks the model to answer using
only those passages. This grounds the answer in real sources and makes it
possible to cite where each claim came from. A typical RAG pipeline has a few
stages. First, documents are split into chunks. Then each chunk is converted
into an embedding and stored in a vector database. At question time, the
question is embedded, the nearest chunks are retrieved, and those chunks are
passed to the language model as context. The quality of the final answer
depends heavily on the quality of retrieval, which in turn depends on how the
documents were chunked and which embedding model was used.
""".strip()


def _demo():
    print("Chunking a sample passage (chunk_size=40 words, overlap=10):\n")
    chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=40, overlap=10)
    for ch in chunks:
        preview = ch.text[:70] + ("..." if len(ch.text) > 70 else "")
        print(f"  chunk {ch.index} (from word {ch.start_word}): {preview}")
    print(f"\nTotal chunks: {len(chunks)}")
    total_words = len(split_words(normalize_whitespace(SAMPLE)))
    print(f"Source words: {total_words}")


if __name__ == "__main__":
    _demo()
