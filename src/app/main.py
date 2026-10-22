"""FastAPI endpoints using the existing FAQ retriever and grounding agent."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from citation_grounding_agent.schemas import GroundedAnswer

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
load_dotenv(APP_DIR / ".env", override=False)

CHUNKS_PATH = PROJECT_ROOT / os.getenv("FAQ_CHUNKS_PATH", "data/faqs/faq_chunks.json")
DB_PATH = PROJECT_ROOT / os.getenv("FAQ_DB_PATH", "data/chroma_embeddinggemma")
AGENT_PATH = PROJECT_ROOT / "src/citation_grounding_agent/grounded_agent.py"
RETRIEVER_LOCK = Lock()
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Grounded FAQ Chat", version="1.0.0")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    mode: Literal["hybrid", "semantic", "bm25"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20, strict=True)


@lru_cache(maxsize=2)
def get_retriever(use_semantic: bool):
    from citation_grounding_agent.hybrid_retriever import HybridRetriever

    # Never create or rebuild an index implicitly in response to a question.
    if use_semantic and not (DB_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            "The Chroma index is missing. Check FAQ_DB_PATH or select BM25."
        )
    retriever = HybridRetriever.from_paths(
        chunks_path=CHUNKS_PATH,
        db_path=DB_PATH,
        device=os.getenv("FAQ_EMBEDDING_DEVICE", "cpu"),
        enable_semantic=use_semantic,
    )
    if use_semantic and not retriever.vector_store.get(limit=1)["ids"]:
        raise ValueError(
            "The Chroma collection is empty. Populate the existing index before using hybrid or semantic search."
        )
    return retriever


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "chat_config": {
                "chat_url": str(request.url_for("chat")),
                "health_url": str(request.url_for("health")),
                "modes": ["hybrid", "semantic", "bm25"],
                "default_mode": "hybrid",
                "top_k": 5,
            }
        },
    )


@app.get("/health")
def health():
    """Check basic setup without loading embeddings or making an API call."""
    try:
        compile(AGENT_PATH.read_text(encoding="utf-8"), str(AGENT_PATH), "exec")
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError(
                "Set OPENAI_API_KEY in src/app/.env or your shell, then restart the server."
            )
        if not CHUNKS_PATH.is_file():
            raise FileNotFoundError(
                "The FAQ chunks file is missing. Check FAQ_CHUNKS_PATH."
            )
        if not (DB_PATH / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                "The default hybrid search needs your existing Chroma index. Check FAQ_DB_PATH."
            )
    except SyntaxError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "setup_required",
                "message": f"The existing grounded_agent.py has a syntax error on line {exc.lineno}. Correct it separately, then retry. The app has not changed it.",
            },
        )
    except (OSError, ValueError) as exc:
        return JSONResponse(
            status_code=503, content={"status": "setup_required", "message": str(exc)}
        )
    return {
        "status": "ready",
        "message": "Basic setup is ready. The retriever and embedding model load on the first request.",
    }


@app.post("/api/chat", response_model=GroundedAnswer)
def chat(payload: ChatRequest):
    # A regular def endpoint runs in FastAPI's worker pool. The shared retriever
    # changes BM25.k per call, so protect both cache initialization and execution.
    try:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise HTTPException(
                503,
                "Set OPENAI_API_KEY in src/app/.env or your shell, then restart the server.",
            )
        from citation_grounding_agent.grounded_agent import ask_grounded_question

        with RETRIEVER_LOCK:
            retriever = get_retriever(payload.mode != "bm25")
            answer = ask_grounded_question(
                payload.question,
                retriever,
                mode=payload.mode,
                k=payload.k,
            )
        return GroundedAnswer.model_validate(answer)
    except HTTPException:
        raise
    except SyntaxError as exc:
        raise HTTPException(
            503,
            f"The existing grounded_agent.py has a syntax error on line {exc.lineno}. Correct it separately before asking a question.",
        ) from exc
    except ImportError as exc:
        LOGGER.exception("Grounding dependencies are unavailable")
        raise HTTPException(
            503,
            "An agent dependency is missing. Run uv sync in the project root and restart the server.",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Grounded FAQ request failed")
        raise HTTPException(
            502,
            "The agent could not complete this request. Check the server log, Chroma index, and model/API configuration, then try again.",
        ) from exc
