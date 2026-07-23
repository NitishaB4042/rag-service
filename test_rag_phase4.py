"""
Tests for RAG Phase 4 — the HTTP service. Drives the real FastAPI app
in-process via httpx ASGITransport (no separate server).
Run: python test_rag_phase4.py
"""
import asyncio
import httpx
import rag_service as svc


async def with_app(body):
    async with svc.app.router.lifespan_context(svc.app):
        transport = httpx.ASGITransport(app=svc.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await body(c)


def test_health_then_ingest_then_ask():
    async def body(c):
        await c.post("/reset")
        h = await c.get("/health")
        assert h.status_code == 200 and h.json()["chunks"] == 0

        ing = await c.post("/ingest", json={"documents": [
            {"text": "The capital of France is Paris. Photosynthesis converts sunlight to sugars.",
             "source": "facts"}], "chunk_size": 8, "overlap": 2})
        assert ing.status_code == 200, ing.text
        assert ing.json()["ingested_chunks"] > 0

        a = await c.post("/ask", json={"question": "What is the capital of France?"})
        assert a.status_code == 200, a.text
        data = a.json()
        assert "Paris" in data["answer"], data["answer"]
        assert data["grounded"] is True
        assert len(data["sources"]) > 0
    asyncio.run(with_app(body))
    print("  health_then_ingest_then_ask: PASS")


def test_ask_before_ingest_409():
    async def body(c):
        await c.post("/reset")
        a = await c.post("/ask", json={"question": "anything?"})
        assert a.status_code == 409, a.status_code
    asyncio.run(with_app(body))
    print("  ask_before_ingest_409: PASS")


def test_refusal_when_not_in_docs():
    async def body(c):
        await c.post("/reset")
        await c.post("/ingest", json={"documents": [
            {"text": "The capital of France is Paris.", "source": "geo"}],
            "chunk_size": 8, "overlap": 2})
        a = await c.post("/ask", json={"question": "boiling point of mercury?"})
        data = a.json()
        assert data["grounded"] is False
        assert data["sources"] == []
    asyncio.run(with_app(body))
    print("  refusal_when_not_in_docs: PASS")


def test_bad_ingest_validation():
    async def body(c):
        await c.post("/reset")
        # empty documents list -> pydantic 422
        r = await c.post("/ingest", json={"documents": []})
        assert r.status_code == 422, r.status_code
        # overlap >= chunk_size -> our explicit 422
        r2 = await c.post("/ingest", json={"documents": [{"text": "hi"}],
                                           "chunk_size": 5, "overlap": 5})
        assert r2.status_code == 422, r2.status_code
    asyncio.run(with_app(body))
    print("  bad_ingest_validation: PASS")


def test_reset_clears_store():
    async def body(c):
        await c.post("/ingest", json={"documents": [{"text": "some text here"}],
                                      "chunk_size": 5, "overlap": 1})
        await c.post("/reset")
        h = await c.get("/health")
        assert h.json()["chunks"] == 0
    asyncio.run(with_app(body))
    print("  reset_clears_store: PASS")


if __name__ == "__main__":
    print("Running RAG Phase 4 tests:")
    test_health_then_ingest_then_ask()
    test_ask_before_ingest_409()
    test_refusal_when_not_in_docs()
    test_bad_ingest_validation()
    test_reset_clears_store()
    print("All tests passed.")
