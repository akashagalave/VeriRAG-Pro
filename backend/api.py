from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
import asyncio

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
from fastapi import Request

logger = logging.getLogger(__name__)

_rate = os.getenv("RATE_LIMIT_PER_MINUTE", "30")
_redis_url = os.getenv("REDIS_URL", "")



def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")

    if xff:
        return xff.split(",")[0].strip()

    return request.client.host or "unknown"
limiter = Limiter(
    key_func=get_client_ip,
    storage_uri=_redis_url if _redis_url else "memory://",
)

DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    _pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(_pg_conn)
    checkpointer.setup() 
    logger.info("Checkpointer: PostgreSQL (shared sessions across all pods)")
else:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    _db_path = os.getenv("CHECKPOINT_DB", "checkpoints.db")
    _sqlite_conn = sqlite3.connect(_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(_sqlite_conn)
    logger.info("Checkpointer: SQLite at %s (local dev — breaks with 2+ pods)", _db_path)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Building LangGraph…")
    _graph = build_graph(checkpointer=checkpointer)
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


app = FastAPI(
    title="VeriRAG API",
    description=(
        "**VeriRAG — Agentic Research Intelligence & Scientific Claim Verification**\n\n"
        "Production Edition with:\n"
        "- Multi-layer GenAI security (prompt injection detection, PII redaction)\n"
        "- Redis distributed document deduplication\n"
        "- Model router with circuit breakers (GPT → Claude → Gemini fallback)\n"
        "- Hybrid BM25 + dense retrieval with RRF fusion\n"
        "- LangSmith tracing for all graph nodes"
    ),
    version="2.1.0",
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
    "/sessions/{session_id}/info",
    response_model=SessionInfoResponse,
    tags=["sessions"],
    summary="Collection stats for a session",
)
async def session_info(session_id: str):
    stats = collection_stats(session_id)
    titles = list_papers(session_id) if stats["exists"] else []
    return SessionInfoResponse(session_id=session_id, paper_titles=titles, **stats)


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

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  

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
@limiter.limit(os.getenv("INGEST_RATE_LIMIT", "5/minute"))
async def ingest(
    request: Request,
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
        docs = None

        if file:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    422, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
                )
            raw_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
            if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    413, f"File exceeds max size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                )

            doc_hash = compute_document_hash(raw_bytes)
            claimed = is_document_indexed(doc_hash, session_id)
            if claimed:
                return IngestResponse(
                    session_id=session_id,
                    message="Document already indexed — skipped (dedup cache hit).",
                    chunks_added=0,
                    deduplicated=True,
                )

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name

                docs = await asyncio.to_thread(load_document, tmp_path)
                for doc in docs:
                    doc.metadata.setdefault("title", Path(file.filename).stem)
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

            await asyncio.to_thread(add_paper, docs, session_id)
            mark_document_indexed(
                doc_hash, session_id,
                chunk_count=len(docs),
                collection_name=get_collection_name(session_id),
            )

        elif url:
            docs = await asyncio.to_thread(load_webpage, url.strip())
            await asyncio.to_thread(add_paper, docs, session_id)

        else:
            docs = await asyncio.to_thread(load_arxiv, arxiv_query.strip())
            await asyncio.to_thread(add_paper, docs, session_id)

        return IngestResponse(
            session_id=session_id,
            message="Documents ingested successfully.",
            chunks_added=len(docs),
            deduplicated=deduplicated,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Ingest failed for session %s", session_id)
        raise HTTPException(500, "Ingest failed. Please try again or contact support.")


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

    # ── Layer 1: Input Guardrail 
    guardrail = check_input(body.question)
    if not guardrail.safe:
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

        full_answer: str = ""
        route: str = "unknown"

        try:
            async for chunk, metadata in graph.stream(
                input_state, config, stream_mode="messages"
            ):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    full_answer += chunk.content
                    yield json.dumps({"type": "token", "data": chunk.content}) + "\n"

            final_values = (await graph.aget_state(config)).values 
            route = final_values.get("route") or "unknown"

            retrieved_docs = final_values.get("retrieved_docs") or []
            sources = [
                {"content": doc.page_content[:500], "metadata": doc.metadata}
                for doc in retrieved_docs
            ]

            if not full_answer:
                full_answer = final_values.get("answer") or ""

            #Layer 3: Output Validator
            validation = validate_output(full_answer)
            if not validation.safe:
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
            yield json.dumps({"type": "error", "data": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")
