import io
import json
import pytest
import requests
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.main import app
import rag_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_test_cache():
    """Ensure clean query cache for test isolation."""
    rag_service.clear_query_cache()

# --- 1. HEALTH ENDPOINT TESTS ---
def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "providers" in data
    assert "groq" in data["providers"]
    assert "sarvam" in data["providers"]
    assert "ollama" in data["providers"]

# --- 2. LANGUAGE LIST ENDPOINT TESTS ---
def test_languages_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    lang_names = [l["name"] for l in data["languages"]]
    assert "English" in lang_names
    assert "हिंदी (Hindi)" in lang_names
    assert "मराठी (Marathi)" in lang_names
    # Verify exact match with rag_service.LANGUAGE_CODES
    for name, code in rag_service.LANGUAGE_CODES.items():
        assert any(l["name"] == name and l["code"] == code for l in data["languages"])

# --- 3. DOCUMENT UPLOAD & REGISTRY TESTS ---
def test_document_upload_validation():
    # Unsupported file extension
    files = [("files", ("malicious.exe", b"binarycontent", "application/octet-stream"))]
    response = client.post("/api/documents", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

    # Empty file
    files = [("files", ("empty.txt", b"", "text/plain"))]
    response = client.post("/api/documents", files=files)
    assert response.status_code == 400
    assert "is empty" in response.json()["detail"]

def test_document_upload_and_caching():
    sample_content = b"DocuMind RAG is a fast, multilingual document retrieval and generation engine."
    
    # First upload (fresh or cache depending on existing disk state)
    files = [("files", ("test_sample.txt", sample_content, "text/plain"))]
    resp1 = client.post("/api/documents", files=files)
    assert resp1.status_code == 200
    docs1 = resp1.json()
    assert len(docs1) == 1
    doc_id = docs1[0]["doc_id"]
    assert docs1[0]["file_name"] == "test_sample.txt"
    assert docs1[0]["chunk_count"] >= 1

    # Second upload of identical content -> MUST hit on-disk cache
    files2 = [("files", ("test_sample.txt", sample_content, "text/plain"))]
    resp2 = client.post("/api/documents", files=files2)
    assert resp2.status_code == 200
    docs2 = resp2.json()
    assert len(docs2) == 1
    assert docs2[0]["cached"] is True
    assert docs2[0]["doc_id"] == doc_id

    # List registered documents
    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    all_docs = list_resp.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in all_docs)

    # Delete registered document
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Confirm deletion from registry
    list_resp_after = client.get("/api/documents")
    assert not any(d["doc_id"] == doc_id for d in list_resp_after.json()["documents"])

    # Deleting non-existent document returns 404
    del_404 = client.delete(f"/api/documents/non_existent_id")
    assert del_404.status_code == 404

# --- 4. CHAT STREAMING TESTS (SSE) ---
def test_chat_stream_empty_question():
    resp = client.post("/api/chat/stream", json={"question": "   ", "language": "English"})
    assert resp.status_code == 400

def test_chat_stream_english():
    # Ingest a small text document first
    sample_txt = b"The capital of Maharashtra is Mumbai. Pune is the cultural capital."
    files = [("files", ("maharashtra.txt", sample_txt, "text/plain"))]
    upload_resp = client.post("/api/documents", files=files)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()[0]["doc_id"]

    # Stream query
    payload = {
        "question": "What is the capital of Maharashtra?",
        "language": "English",
        "doc_ids": [doc_id]
    }
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        lines = list(response.iter_lines())
        tokens = []
        final_event = None
        done_seen = False

        for line in lines:
            if not line:
                continue
            if line.startswith("data: "):
                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    done_seen = True
                    continue
                try:
                    event = json.loads(raw_data)
                    if event.get("type") == "token":
                        tokens.append(event.get("content", ""))
                    elif event.get("type") == "final":
                        final_event = event
                except Exception:
                    pass

        assert done_seen is True
        assert len(tokens) > 0
        assert final_event is not None
        assert "citations" in final_event
        assert "metrics" in final_event
        assert "retrieval_ms" in final_event["metrics"]
        assert "source" in final_event

    # Cleanup
    client.delete(f"/api/documents/{doc_id}")

# --- 5. MULTILINGUAL STREAMING TEST (HINDI) ---
def test_chat_stream_hindi():
    sample_txt = b"Artificial intelligence is transforming software engineering rapidly."
    files = [("files", ("ai.txt", sample_txt, "text/plain"))]
    upload_resp = client.post("/api/documents", files=files)
    doc_id = upload_resp.json()[0]["doc_id"]

    payload = {
        "question": "What is transforming software engineering?",
        "language": "हिंदी (Hindi)",
        "doc_ids": [doc_id]
    }
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        done_seen = False
        final_event = None

        for line in lines:
            if not line:
                continue
            if line.startswith("data: "):
                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    done_seen = True
                else:
                    try:
                        ev = json.loads(raw_data)
                        if ev.get("type") == "final":
                            final_event = ev
                    except Exception:
                        pass

        assert done_seen is True
        assert final_event is not None
        assert final_event["metrics"]["target_lang"] == "हिंदी (Hindi)"
        assert "translation_tiers" in final_event["metrics"]

    client.delete(f"/api/documents/{doc_id}")

# --- 6. FALLBACK PATH MOCK TESTS ---
def test_fallback_when_groq_and_ollama_fail():
    """Verify that when both Groq and Ollama fail, heuristic scoring gracefully succeeds."""
    sample_txt = b"Quantum computing uses qubits to represent superposition and entanglement."
    files = [("files", ("quantum.txt", sample_txt, "text/plain"))]
    upload_resp = client.post("/api/documents", files=files)
    doc_id = upload_resp.json()[0]["doc_id"]

    # Mock Groq to raise exception and Ollama to raise exception
    with patch("rag_service.call_groq_api", side_effect=RuntimeError("Groq API rate limit or outage")), \
         patch("rag_service.HTTP_SESSION.post", side_effect=requests.exceptions.ConnectionError("Ollama down")):
        
        payload = {
            "question": "What does quantum computing use?",
            "language": "English",
            "doc_ids": [doc_id]
        }
        with client.stream("POST", "/api/chat/stream", json=payload) as response:
            assert response.status_code == 200
            final_event = None
            for line in response.iter_lines():
                if line and line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        ev = json.loads(line[6:].strip())
                        if ev.get("type") == "final":
                            final_event = ev
                    except Exception:
                        pass

            assert final_event is not None
            assert "Heuristic Scoring" in final_event["source"]

    client.delete(f"/api/documents/{doc_id}")

def test_translation_fallback_tier3():
    """Verify that when Sarvam and LLM translation fail, Tier 3 returns text with note."""
    sample_txt = b"Solar energy is renewable and reduces carbon emissions."
    files = [("files", ("solar.txt", sample_txt, "text/plain"))]
    upload_resp = client.post("/api/documents", files=files)
    doc_id = upload_resp.json()[0]["doc_id"]

    with patch("rag_service.get_sarvam_api_key", return_value="fake_sarvam_key"), \
         patch("rag_service.translate_with_sarvam", side_effect=RuntimeError("Sarvam service down")), \
         patch("rag_service.translate_with_llm", side_effect=RuntimeError("LLM translation failed")):

        payload = {
            "question": "What is solar energy?",
            "language": "मराठी (Marathi)",
            "doc_ids": [doc_id]
        }
        with client.stream("POST", "/api/chat/stream", json=payload) as response:
            assert response.status_code == 200
            final_event = None
            for line in response.iter_lines():
                if line and line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        ev = json.loads(line[6:].strip())
                        if ev.get("type") == "final":
                            final_event = ev
                    except Exception:
                        pass

            assert final_event is not None
            assert any("Translation Unavailable" in t for t in final_event["metrics"]["translation_tiers"]) or "Translation Unavailable" in final_event["source"]

    client.delete(f"/api/documents/{doc_id}")
