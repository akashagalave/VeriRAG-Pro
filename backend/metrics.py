"""
backend/metrics.py
------------------
Prometheus instrumentation for VeriRAG.

Exposed at:  GET /metrics   (scraped by Prometheus every 15s)
Visualised in Grafana dashboard (k8s/monitoring.yml).

Metrics tracked:
  Counters:    request_count, error_count, token_usage, cache_hits/misses, ingest_count
  Histograms:  request_latency (p50/p95/p99), llm_latency per node, retrieval_latency
  Gauges:      circuit_breaker_open per provider

Fail-open design:
  If prometheus_client is not installed, all functions are no-ops.
  The application never crashes because of missing metrics.

Usage:
  from backend.metrics import (
      record_request, inc_error, record_ingest,
      record_cache_hit, record_cache_miss,
      measure_request, measure_llm, measure_retrieval,
      get_metrics_output,
  )
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

# ── Optional prometheus import ────────────────────────────────────────────────

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.info(
        "prometheus_client not installed — metrics disabled. "
        "pip install prometheus-client to enable."
    )

# ── Token cost table (USD per 1M tokens, approximate May 2026) ────────────────

_COST_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4o":       {"input": 5.00,  "output": 15.00},
    "claude-haiku": {"input": 0.25,  "output": 1.25},
    "gemini-flash": {"input": 0.075, "output": 0.30},
}

# ── Metric definitions ────────────────────────────────────────────────────────

if _AVAILABLE:

    # ── Counters ──────────────────────────────────────────────────────────────

    REQUEST_COUNT = Counter(
        "verirag_request_count_total",
        "Total RAG query requests received",
        ["route"],
    )
    ERROR_COUNT = Counter(
        "verirag_error_count_total",
        "Total errors by type",
        ["error_type"],
    )
    TOKEN_USAGE = Counter(
        "verirag_token_usage_total",
        "Total LLM tokens consumed",
        ["model", "token_type"],   # token_type: input | output
    )
    CACHE_HIT_COUNT = Counter(
        "verirag_cache_hit_total",
        "Redis document deduplication cache hits (documents skipped on re-ingest)",
    )
    CACHE_MISS_COUNT = Counter(
        "verirag_cache_miss_total",
        "Redis document deduplication cache misses (new documents indexed)",
    )
    INGEST_COUNT = Counter(
        "verirag_ingest_count_total",
        "Documents ingested by source type",
        ["source_type"],           # file | url | arxiv
    )
    SECURITY_BLOCK_COUNT = Counter(
        "verirag_security_block_total",
        "Requests blocked by security layer",
        ["layer"],                 # input_guardrail | output_validator
    )

    # ── Histograms ────────────────────────────────────────────────────────────

    REQUEST_LATENCY = Histogram(
        "verirag_request_latency_seconds",
        "End-to-end request latency (router → generate_answer → stream done)",
        ["route"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    )
    LLM_LATENCY = Histogram(
        "verirag_llm_latency_seconds",
        "LLM call latency per graph node",
        ["node", "model"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    RETRIEVAL_LATENCY = Histogram(
        "verirag_retrieval_latency_seconds",
        "Qdrant hybrid retrieval latency (BM25+dense+RRF)",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )
    COST_PER_REQUEST = Histogram(
        "verirag_estimated_cost_usd",
        "Estimated USD cost per query request (input + output tokens)",
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.50],
    )

    # ── Gauges ────────────────────────────────────────────────────────────────

    CIRCUIT_BREAKER_OPEN = Gauge(
        "verirag_circuit_breaker_open",
        "1 = circuit open (provider failing/blocked), 0 = closed (healthy)",
        ["provider"],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def record_request(route: str) -> None:
    """Increment request counter for a given route."""
    if not _AVAILABLE:
        return
    REQUEST_COUNT.labels(route=route or "unknown").inc()


def inc_error(error_type: str) -> None:
    """Increment error counter by type (e.g. 'stream_error', 'graph_error')."""
    if not _AVAILABLE:
        return
    ERROR_COUNT.labels(error_type=error_type).inc()


def record_security_block(layer: str) -> None:
    """Increment security block counter (layer = 'input_guardrail' | 'output_validator')."""
    if not _AVAILABLE:
        return
    SECURITY_BLOCK_COUNT.labels(layer=layer).inc()


def record_cache_hit() -> None:
    if not _AVAILABLE:
        return
    CACHE_HIT_COUNT.inc()


def record_cache_miss() -> None:
    if not _AVAILABLE:
        return
    CACHE_MISS_COUNT.inc()


def record_ingest(source_type: str) -> None:
    """source_type: 'file' | 'url' | 'arxiv'"""
    if not _AVAILABLE:
        return
    INGEST_COUNT.labels(source_type=source_type).inc()


def record_token_usage(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """
    Record token consumption and compute estimated USD cost.
    Call this after any LLM invocation where token counts are available.
    """
    if not _AVAILABLE:
        return
    if input_tokens:
        TOKEN_USAGE.labels(model=model, token_type="input").inc(input_tokens)
    if output_tokens:
        TOKEN_USAGE.labels(model=model, token_type="output").inc(output_tokens)

    costs = _COST_PER_MILLION.get(model, {"input": 0.15, "output": 0.60})
    estimated_cost = (
        input_tokens  * costs["input"] +
        output_tokens * costs["output"]
    ) / 1_000_000
    if estimated_cost > 0:
        COST_PER_REQUEST.observe(estimated_cost)


def update_circuit_breakers() -> None:
    """
    Sync circuit breaker states into Prometheus gauges.
    Called from get_metrics_output() so gauges are always fresh at scrape time.
    """
    if not _AVAILABLE:
        return
    try:
        from backend.model_router import get_circuit_breaker_status
        for status in get_circuit_breaker_status():
            CIRCUIT_BREAKER_OPEN.labels(provider=status["provider"]).set(
                1.0 if status["circuit_state"] == "open" else 0.0
            )
    except Exception as exc:
        logger.debug("update_circuit_breakers: %s", exc)


# ── Context managers ──────────────────────────────────────────────────────────

@contextmanager
def measure_request(route: str) -> Generator[None, None, None]:
    """
    Time a full end-to-end request.

    Usage:
        with measure_request(route="retrieve"):
            ... run graph ...
    """
    if not _AVAILABLE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        REQUEST_LATENCY.labels(route=route or "unknown").observe(
            time.perf_counter() - start
        )


@contextmanager
def measure_llm(node: str, model: str = "gpt-4o-mini") -> Generator[None, None, None]:
    """
    Time an LLM call within a specific LangGraph node.

    Usage:
        with measure_llm(node="router", model="gpt-4o-mini"):
            result = llm.invoke(messages)
    """
    if not _AVAILABLE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        LLM_LATENCY.labels(node=node, model=model).observe(
            time.perf_counter() - start
        )


@contextmanager
def measure_retrieval() -> Generator[None, None, None]:
    """
    Time a Qdrant hybrid retrieval call.

    Usage:
        with measure_retrieval():
            docs = vs_search(query, session_id, k)
    """
    if not _AVAILABLE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start)


# ── Scrape endpoint helper ────────────────────────────────────────────────────

def get_metrics_output() -> tuple[bytes, str]:
    """
    Generate Prometheus text-format metrics for GET /metrics.
    Updates circuit breaker gauges before generating output.

    Returns:
        (metrics_bytes, content_type_string)
    """
    if not _AVAILABLE:
        return (
            b"# prometheus_client not installed. pip install prometheus-client\n",
            "text/plain; charset=utf-8",
        )
    update_circuit_breakers()
    return generate_latest(), CONTENT_TYPE_LATEST
