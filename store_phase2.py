"""
Semantic Search + RAG Service — Phase 2: embed, store, and search.

Turns the chunks from Phase 1 into EMBEDDINGS (vectors that capture meaning),
stores them in a vector database (Chroma), and supports semantic SEARCH:
embed a question, find the nearest chunks by meaning.

Design choice: the embedder is PLUGGABLE behind a tiny interface, so the store
and search code never depends on one provider. Two embedders ship here:

  - HashingEmbedder       : zero heavy dependencies. Deterministic, good for
                            wiring up and testing the pipeline anywhere
                            (including a tablet sandbox). NOT semantically smart.
  - SentenceTransformerEmbedder : the real one for Colab / a real machine.
                            Uncomment the import; produces genuine semantic
                            embeddings. Same interface, so nothing else changes.

Run the demo:   python store_phase2.py
Run the tests:  python test_phase2.py
"""

import os
import math
import hashlib
from dataclasses import dataclass

import chromadb

from ingest_phase1 import Chunk, chunk_text


# ===========================================================================
# Embedder interface — anything with .name and .embed(list[str]) -> list[vec]
# ===========================================================================
class Embedder:
    name = "base"
    dim = 0
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """A dependency-free embedder for wiring up and testing the pipeline.

    It maps words to fixed positions in a vector via hashing (the "hashing
    trick") and L2-normalizes the result. Texts that share words land near
    each other, so it demonstrates the mechanics of semantic search end to end.
    It is NOT a real language model \u2014 swap in SentenceTransformerEmbedder for
    genuine meaning-based matching. The rest of the system is unchanged.
    """
    name = "hashing-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        # L2-normalize so cosine similarity behaves well
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# Real embedder for Colab / a real machine. Uncomment to use:
#
# from sentence_transformers import SentenceTransformer
# class SentenceTransformerEmbedder(Embedder):
#     name = "all-MiniLM-L6-v2"
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.dim = self.model.get_sentence_embedding_dimension()
#     def embed(self, texts):
#         return self.model.encode(texts, normalize_embeddings=True).tolist()


# ===========================================================================
# Vector store wrapper (Chroma)
# ===========================================================================
class VectorStore:
    """Stores chunk embeddings and supports nearest-neighbour search."""

    def __init__(self, embedder: Embedder, collection_name: str | None = None,
                 persist_dir: str | None = None):
        self.embedder = embedder
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()   # in-memory, great for tests
        # a unique name per store unless caller pins one, so separate stores
        # never accidentally share a collection in the same process
        if collection_name is None:
            import uuid
            collection_name = f"rag_{uuid.uuid4().hex[:12]}"
        # cosine space matches our normalized embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        ids = [f"{c.source}:{c.index}" for c in chunks]
        metadatas = [{"source": c.source, "index": c.index,
                      "start_word": c.start_word} for c in chunks]
        self.collection.add(ids=ids, embeddings=vectors,
                            documents=texts, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k most similar chunks to `query`.

        Each result: {text, source, index, score} where score is similarity
        (1.0 = identical direction, higher = more similar).
        """
        qvec = self.embedder.embed([query])[0]
        res = self.collection.query(query_embeddings=[qvec], n_results=k)
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "text": doc,
                "source": meta["source"],
                "index": meta["index"],
                "score": 1.0 - dist,        # cosine distance -> similarity
            })
        return out

    def count(self) -> int:
        return self.collection.count()


# ===========================================================================
# Demo
# ===========================================================================
SAMPLE = """
Retrieval-augmented generation answers questions using a language model
together with your own documents. The system retrieves relevant passages and
then asks the model to answer using only those passages, which grounds the
answer in real sources. Documents are first split into chunks. Each chunk is
turned into an embedding and stored in a vector database. At question time, the
question is embedded and the nearest chunks are retrieved. Cooking pasta well
requires salting the water generously and not overcooking the noodles. A good
tomato sauce simmers slowly with garlic and basil. Photosynthesis lets plants
convert sunlight into chemical energy stored as sugars.
""".strip()


def _demo():
    embedder = HashingEmbedder()
    store = VectorStore(embedder)
    chunks = chunk_text(SAMPLE, source="mixed.txt", chunk_size=25, overlap=5)
    n = store.add(chunks)
    print(f"Embedder: {embedder.name} (dim {embedder.dim})")
    print(f"Stored {n} chunks. Total in store: {store.count()}\n")

    for q in ["how are documents prepared for retrieval?",
              "what do plants do with sunlight?",
              "how do I make a good pasta sauce?"]:
        print(f"Q: {q}")
        for r in store.search(q, k=2):
            preview = r["text"][:65] + ("..." if len(r["text"]) > 65 else "")
            print(f"   [score {r['score']:.2f}] {preview}")
        print()

    print("Note: this demo uses the dependency-free HashingEmbedder, which "
          "matches on shared words. On Colab, switch to "
          "SentenceTransformerEmbedder for true meaning-based matching.")


if __name__ == "__main__":
    _demo()
