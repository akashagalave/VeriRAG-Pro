"""
backend/api.py
--------------
FastAPI backend for VeriRAG — Production Edition.

Routes
~~~~~~
  GET  /health                             — liveness probe
  GET  /ready                              — readiness probe
  GET  /metrics                            — Prometheus scrape endpoint (NEW)
  POST /sessions/{session_id}/ingest       — load docs (with Redis dedup)    (UPDATED)
  GET  /sessions/{session_id}/info         — collection stats
  GET  /sessions/{session_id}/history      — reload past chat messages
  POST /sessions/{session_id}/query        — RAG pipeline + security layers  (UPDATED)

Production additions vs original:
  Security Layer   — Input guardrail (Layer 1) before graph; Output validator
                     (Layer 3) before streaming done event.
  Redis Dedup      — SHA-256 document hash check before every ingest.
  Prometheus       — /metrics endpoint; request count, error count per route.
  Model Router     — Circuit breaker status exposed in /health.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.metrics import (
    get_metrics_output,
    inc_error,
    measure_request,
    record_ingest,
    record_cache_hit,
    record_cache_miss,
    record_request,
    record_security_block,
)
from backend.model_router import get_circuit_breaker_status
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.redis_cache import (
    compute_document_hash,
    is_document_indexed,
    mark_document_indexed,
)
from backend.security import check_input, validate_output
from backend.vector_store import add_paper, collection_stats, get_collection_name, list_papers

logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate = os.getenv("RATE_LIMIT_PER_MINUTE", "30")
limiter = Limiter(key_func=get_remote_address)

# ── LangGraph singleton ───────────────────────────────────────────────────────

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Building LangGraph…")
    _graph = build_graph(db_path=os.getenv("CHECKPOINT_DB", "checkpoints.db"))
    logger.info("LangGraph ready.")
    yield
    logger.info("VeriRAG FastAPI shutdown.")


def get_graph():
    if _graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialised yet — retry in a moment.",
        )
    return _graph


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VeriRAG API",
    description=(
        "**VeriRAG — Agentic Research Intelligence & Scientific Claim Verification**\n\n"
        "Production Edition with:\n"
        "- Multi-layer GenAI security (prompt injection detection, PII redaction)\n"
        "- Redis distributed document deduplication\n"
        "- Model router with circuit breakers (GPT → Claude → Gemini fallback)\n"
        "- Prometheus metrics at `/metrics`\n"
        "- Hybrid BM25 + dense retrieval with RRF fusion\n"
        "- LangSmith tracing for all graph nodes"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="Natural-language question, claim to verify, or general query.",
    )


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[list] = None
    route: Optional[str] = None


class IngestResponse(BaseModel):
    session_id: str
    message: str
    chunks_added: int
    deduplicated: bool = False


class SessionInfoResponse(BaseModel):
    session_id: str
    exists: bool
    chunk_count: int
    hybrid: bool
    collection_name: Optional[str] = None
    paper_titles: list[str]


# ── Health / readiness / metrics ──────────────────────────────────────────────

@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health():
    """Always 200. Includes circuit breaker status for each LLM provider."""
    return {
        "status": "ok",
        "providers": get_circuit_breaker_status(),
    }


@app.get("/ready", tags=["ops"], summary="Readiness probe")
async def ready():
    """Returns 200 only after LangGraph has compiled."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready yet.")
    return {"status": "ready"}


@app.get(
    "/metrics",
    tags=["ops"],
    summary="Prometheus metrics scrape endpoint",
    description=(
        "Returns Prometheus text-format metrics.\n\n"
        "Configure Prometheus to scrape this endpoint every 15s:\n"
        "```yaml\n"
        "scrape_configs:\n"
        "  - job_name: verirag\n"
        "    static_configs:\n"
        "      - targets: ['verirag-fastapi-service:8000']\n"
        "```"
    ),
)
async def metrics():
    data, content_type = get_metrics_output()
    return Response(content=data, media_type=content_type)


# ── Session info ──────────────────────────────────────────────────────────────

@app.get(
    "/sessions/{session_id}/info",
    response_model=SessionInfoResponse,
    tags=["sessions"],
    summary="Collection stats for a session",
)
async def session_info(session_id: str):
    stats = collection_stats(session_id)
    titles = list_papers(session_id) if stats["exists"] else []
    return SessionInfoResponse(session_id=session_id, paper_titles=titles, **stats)


# ── Session history ───────────────────────────────────────────────────────────

@app.get(
    "/sessions/{session_id}/history",
    response_model=list[ChatMessage],
    tags=["sessions"],
    summary="Reload chat history for a session",
)
async def session_history(session_id: str):
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
    except Exception:
        return []

    messages = state.values.get("messages", [])
    chats: list[ChatMessage] = []
    for msg in messages:
        type_name = type(msg).__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if type_name == "HumanMessage":
            chats.append(ChatMessage(role="user", content=content))
        elif type_name in ("AIMessage", "AIMessageChunk"):
            if not content.strip():
                continue
            chats.append(ChatMessage(
                role="assistant", content=content,
                sources=[], route=state.values.get("route"),
            ))
    return chats


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post(
    "/sessions/{session_id}/ingest",
    response_model=IngestResponse,
    tags=["sessions"],
    summary="Ingest a document into a session",
    description=(
        "Accepts a file upload (PDF / TXT / MD), a URL, or an arXiv ID / title.\n\n"
        "**Redis deduplication**: SHA-256 of raw bytes is checked before embedding. "
        "Duplicate documents are skipped — no redundant OpenAI calls across pods."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    session_id: str,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    arxiv_query: Optional[str] = Form(default=None),
):
    provided = sum([file is not None, bool(url), bool(arxiv_query)])
    if provided == 0:
        raise HTTPException(422, "Provide exactly one of: file, url, arxiv_query.")
    if provided > 1:
        raise HTTPException(422, "Only one of file / url / arxiv_query may be supplied.")

    try:
        deduplicated = False

        if file:
            raw_bytes = await file.read()
            doc_hash = compute_document_hash(raw_bytes)

            # ── Redis deduplication check ─────────────────────────────────────
            if is_document_indexed(doc_hash, session_id):
                record_cache_hit()
                return IngestResponse(
                    session_id=session_id,
                    message="Document already indexed — skipped (dedup cache hit).",
                    chunks_added=0,
                    deduplicated=True,
                )
            record_cache_miss()

            suffix = Path(file.filename).suffix
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                docs = load_document(tmp_path)
                for doc in docs:
                    doc.metadata.setdefault("title", Path(file.filename).stem)
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

            add_paper(docs, session_id)
            mark_document_indexed(
                doc_hash, session_id,
                chunk_count=len(docs),
                collection_name=get_collection_name(session_id),
            )
            record_ingest("file")

        elif url:
            docs = load_webpage(url.strip())
            add_paper(docs, session_id)
            record_ingest("url")

        else:
            docs = load_arxiv(arxiv_query.strip())
            add_paper(docs, session_id)
            record_ingest("arxiv")

        return IngestResponse(
            session_id=session_id,
            message="Documents ingested successfully.",
            chunks_added=len(docs),
            deduplicated=deduplicated,
        )

    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Ingest failed for session %s", session_id)
        inc_error("ingest_error")
        raise HTTPException(500, f"Ingest failed: {exc}")


# ── Query (streaming) ─────────────────────────────────────────────────────────

@app.post(
    "/sessions/{session_id}/query",
    tags=["query"],
    summary="Run the RAG pipeline (streaming)",
    description=(
        "Streams the answer token-by-token.\n\n"
        "**Security**: Input is checked for prompt injection before the graph runs. "
        "Output is scanned for PII before the `done` event is sent.\n\n"
        "**Stream protocol** (newline-delimited JSON):\n"
        "```\n"
        "{\"type\": \"token\", \"data\": \"text chunk\"}\n"
        "{\"type\": \"done\",  \"data\": {\"answer\": \"...\", \"route\": \"retrieve\", \"sources\": [...]}}\n"
        "{\"type\": \"error\", \"data\": \"error message\"}\n"
        "```"
    ),
    response_class=StreamingResponse,
)
@limiter.limit(f"{_rate}/minute")
async def query_session(
    request: Request,
    session_id: str,
    body: QueryRequest,
):
    graph = get_graph()

    # ── Layer 1: Input Guardrail (before graph, fail-closed) ──────────────────
    guardrail = check_input(body.question)
    if not guardrail.safe:
        record_security_block("input_guardrail")
        inc_error("prompt_injection_blocked")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=guardrail.reason,
        )

    async def _stream() -> AsyncIterator[str]:
        config = {"configurable": {"thread_id": session_id}}
        input_state = {
            "messages": [HumanMessage(content=body.question)],
            "session_id": session_id,
            "query": body.question,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }

        route: str = "unknown"
        full_answer: str = ""

        try:
            with measure_request(route="pending"):  # updated below
                for chunk, metadata in graph.stream(
                    input_state, config, stream_mode="messages"
                ):
                    if (
                        metadata.get("langgraph_node") == "generate_answer"
                        and hasattr(chunk, "content")
                        and chunk.content
                    ):
                        full_answer += chunk.content
                        yield json.dumps({"type": "token", "data": chunk.content}) + "\n"

            # Read final state
            final_values = graph.get_state(config).values
            route = final_values.get("route") or "unknown"
            record_request(route)

            retrieved_docs = final_values.get("retrieved_docs") or []
            sources = [
                {"content": doc.page_content[:500], "metadata": doc.metadata}
                for doc in retrieved_docs
            ]

            # Ensure we have the full answer from state (streaming may be partial)
            if not full_answer:
                full_answer = final_values.get("answer") or ""

            # ── Layer 3: Output Validator (before done event) ─────────────────
            validation = validate_output(full_answer)
            if not validation.safe:
                record_security_block("output_validator")
                logger.warning(
                    "PII redacted in output for session %s — types: %s",
                    session_id, validation.pii_types,
                )
            safe_answer = validation.redacted_output or full_answer

            yield json.dumps({
                "type": "done",
                "data": {
                    "answer": safe_answer,
                    "route": route,
                    "sources": sources,
                },
            }) + "\n"

        except Exception as exc:
            logger.exception("Stream failed for session %s", session_id)
            inc_error("stream_error")
            yield json.dumps({"type": "error", "data": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")
