"""HTTP-boundary checks; no model API calls or embedding downloads."""

import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from app import main
from citation_grounding_agent.schemas import (
    GroundedAnswer,
    SourceCitation,
    VerifiedClaim,
)


@pytest.fixture
def client():
    with TestClient(main.app) as session:
        yield session


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    calls = []
    answer = GroundedAnswer(
        status="answered",
        evidence_status="supported",
        answer="Employees receive 20 working days of annual leave. [S1]",
        claims=[
            VerifiedClaim(
                statement="Employees receive 20 working days of annual leave.",
                citation_ids=["S1"],
            )
        ],
        citations=[
            SourceCitation(
                citation_id="S1",
                chunk_id="chunk-1",
                faq_id="FAQ-1",
                document_id="DOC-1",
                source_path="pdfs/leave.pdf",
                source_page=1,
                source_version="1.0",
                quote="A full-time employee receives 20 working days of paid annual leave per calendar year.",
            )
        ],
    )
    module = ModuleType("citation_grounding_agent.grounded_agent")

    def ask(question, retriever, **kwargs):
        calls.append({"question": question, "retriever": retriever, **kwargs})
        return answer

    module.ask_grounded_question = ask
    monkeypatch.setitem(sys.modules, module.__name__, module)
    retriever_modes = []

    def retriever(use_semantic):
        retriever_modes.append(use_semantic)
        return "test-retriever"

    monkeypatch.setattr(main, "get_retriever", retriever)
    return module, answer, calls, retriever_modes


def test_template_static_files_and_api_docs(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "@tailwindcss/browser@4" in response.text
    assert '"default_mode": "hybrid"' in response.text
    assert '"modes": ["hybrid", "semantic", "bm25"]' in response.text
    assert client.get("/static/chat.js").status_code == 200
    assert client.get("/static/chat.css").status_code == 200
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize(
    "mode,use_semantic", [("hybrid", True), ("semantic", True), ("bm25", False)]
)
def test_forwards_search_parameters_and_preserves_citations(
    client, fake_agent, mode, use_semantic
):
    _, answer, calls, retriever_modes = fake_agent
    response = client.post(
        "/api/chat", json={"question": "  How much leave?  ", "mode": mode, "k": 8}
    )
    assert response.status_code == 200
    assert response.json() == answer.model_dump()
    assert calls == [
        {
            "question": "How much leave?",
            "retriever": "test-retriever",
            "mode": mode,
            "k": 8,
        }
    ]
    assert retriever_modes == [use_semantic]


def test_defaults_to_hybrid_and_five_sources(client, fake_agent):
    _, _, calls, _ = fake_agent
    assert (
        client.post("/api/chat", json={"question": "How much leave?"}).status_code
        == 200
    )
    assert calls[0]["mode"] == "hybrid"
    assert calls[0]["k"] == 5


def test_preserves_insufficient_evidence(client, fake_agent):
    module, _, _, _ = fake_agent
    answer = GroundedAnswer(
        status="insufficient_evidence",
        evidence_status="low",
        answer="Not enough verified evidence.",
        claims=[],
        citations=[],
        reason="No relevant FAQ.",
    )
    module.ask_grounded_question = lambda *args, **kwargs: answer
    response = client.post("/api/chat", json={"question": "An unsupported question"})
    assert response.status_code == 200
    assert response.json() == answer.model_dump()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 4001},
        {"question": "test", "mode": "other"},
        {"question": "test", "k": 0},
        {"question": "test", "k": 21},
        {"question": "test", "k": "5"},
        {"question": "test", "k": True},
        {"question": "test", "history": []},
    ],
)
def test_rejects_invalid_requests_before_calling_agent(client, fake_agent, payload):
    _, _, calls, _ = fake_agent
    assert client.post("/api/chat", json=payload).status_code == 422
    assert calls == []


def test_missing_key_is_actionable(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"question": "How much leave?"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_runtime_error_does_not_expose_exception_details(client, fake_agent):
    module, _, _, _ = fake_agent

    def failing_agent(*args, **kwargs):
        raise RuntimeError("private-upstream-diagnostic")

    module.ask_grounded_question = failing_agent
    response = client.post("/api/chat", json={"question": "How much leave?"})
    assert response.status_code == 502
    assert "private-upstream-diagnostic" not in response.text


def test_health_reports_syntax_error_without_loading_agent(
    client, monkeypatch, tmp_path
):
    source = tmp_path / "agent.py"
    source.write_text("def broken(\n", encoding="utf-8")
    monkeypatch.setattr(main, "AGENT_PATH", source)
    response = client.get("/health")
    assert response.status_code == 503
    assert "syntax error" in response.json()["message"]
    assert client.get("/").status_code == 200


def test_health_ready_without_loading_embeddings(client, monkeypatch, tmp_path):
    source = tmp_path / "agent.py"
    source.write_text("pass\n", encoding="utf-8")
    chunks = tmp_path / "chunks.json"
    chunks.write_text("{}", encoding="utf-8")
    (tmp_path / "chroma.sqlite3").touch()
    monkeypatch.setattr(main, "AGENT_PATH", source)
    monkeypatch.setattr(main, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(main, "DB_PATH", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    assert client.get("/health").json()["status"] == "ready"
