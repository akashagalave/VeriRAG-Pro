"""
backend/redis_cache.py
----------------------
Redis-backed distributed document deduplication for VeriRAG.

Problem this solves:
  In EKS with 2-5 FastAPI pods, LocalFileStore is per-pod:
    Pod-1 caches chunk embeddings → Pod-1 filesystem
    Pod-2 knows nothing → re-embeds same chunks → wasted OpenAI cost
    Pod restart → all cached embeddings lost

Solution:
  Before ingesting a document, compute SHA-256 of its raw bytes.
  Look up the hash in Redis (shared across all pods and restarts).
  If found → document already indexed for this session → skip pipeline.
  If not  → run chunk/embed/store pipeline → record hash in Redis.

Redis key schema:
  verirag:doc:{session_id}:{sha256_hex}  →  JSON payload
    {
      "doc_hash": "abc123...",
      "session_id": "uuid",
      "collection_name": "papeer_uuid",
      "chunk_count": 42,
      "embedding_version": "text-embedding-3-small-v1",
      "indexed_at": "2026-05-13T12:00:00Z"
    }

TTL: 30 days (configurable via REDIS_DOC_TTL_DAYS env var)

Fail-open design:
  If Redis is unavailable (no REDIS_URL, connection timeout, any error),
  is_document_indexed() returns False (re-index to be safe).
  mark_document_indexed() logs a warning and silently continues.
  The rest of the ingestion pipeline is NEVER blocked by Redis.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_VERSION = "text-embedding-3-small-v1"
KEY_PREFIX = "verirag:doc"
_TTL_DAYS = int(os.getenv("REDIS_DOC_TTL_DAYS", "30"))
TTL_SECONDS = _TTL_DAYS * 24 * 3600

# ── Singleton client ──────────────────────────────────────────────────────────

_client = None
_client_initialized = False


def _get_client():
    """
    Lazy-initialise Redis client on first call.
    Returns None if REDIS_URL is not set or connection fails.
    Subsequent calls return the cached client (or None if init failed).
    """
    global _client, _client_initialized

    if _client_initialized:
        return _client

    _client_initialized = True
    redis_url = os.getenv("REDIS_URL", "")

    if not redis_url:
        logger.info(
            "REDIS_URL not configured — document deduplication cache disabled. "
            "Set REDIS_URL=redis://... to enable distributed caching."
        )
        return None

    try:
        import redis  # type: ignore

        r = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        r.ping()
        _client = r
        logger.info("Redis cache connected: url=%s  ttl=%d days", redis_url, _TTL_DAYS)
    except ImportError:
        logger.warning(
            "redis package not installed — pip install redis. "
            "Document deduplication cache disabled."
        )
    except Exception as exc:
        logger.warning(
            "Redis unavailable (%s) — document deduplication cache disabled. "
            "Ingestion will continue without deduplication.", exc,
        )

    return _client


# ── Core public API ───────────────────────────────────────────────────────────

def compute_document_hash(raw_bytes: bytes) -> str:
    """
    Compute SHA-256 of raw document bytes.
    This is the deduplication key — content-addressable, not filename-based.
    Same file uploaded twice (different name) → same hash → deduplicated.
    """
    return hashlib.sha256(raw_bytes).hexdigest()


def _make_key(session_id: str, doc_hash: str) -> str:
    return f"{KEY_PREFIX}:{session_id}:{doc_hash}"


def is_document_indexed(doc_hash: str, session_id: str) -> bool:
    """
    Check whether this document has already been ingested for this session.

    Returns:
        True  → skip ingestion (already indexed)
        False → proceed with ingestion (not indexed, or Redis unavailable)

    Policy: FAIL OPEN — if Redis errors, return False (re-index to be safe).
    """
    client = _get_client()
    if client is None:
        return False

    try:
        key = _make_key(session_id, doc_hash)
        exists = client.exists(key) == 1
        if exists:
            logger.info(
                "CACHE HIT — document already indexed | session=%s | hash=%s...",
                session_id[:8], doc_hash[:12],
            )
        return exists
    except Exception as exc:
        logger.warning("Redis dedup check failed (FAIL OPEN): %s", exc)
        return False


def mark_document_indexed(
    doc_hash: str,
    session_id: str,
    chunk_count: int,
    collection_name: str,
) -> None:
    """
    Record that a document has been successfully indexed.
    Called AFTER add_paper() succeeds — never before.

    Policy: FAIL OPEN — silently logs and continues on any Redis error.
    """
    client = _get_client()
    if client is None:
        return

    try:
        key = _make_key(session_id, doc_hash)
        payload = json.dumps({
            "doc_hash": doc_hash,
            "session_id": session_id,
            "collection_name": collection_name,
            "chunk_count": chunk_count,
            "embedding_version": EMBEDDING_VERSION,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        })
        client.setex(key, TTL_SECONDS, payload)
        logger.info(
            "CACHE WRITE — marked indexed | session=%s | hash=%s... | chunks=%d",
            session_id[:8], doc_hash[:12], chunk_count,
        )
    except Exception as exc:
        logger.warning("Redis mark_indexed failed (FAIL OPEN): %s", exc)


def get_indexed_metadata(doc_hash: str, session_id: str) -> Optional[dict]:
    """
    Return the stored metadata dict for a cached document, or None.
    Useful for diagnostics and the /info endpoint.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        key = _make_key(session_id, doc_hash)
        raw = client.get(key)
        if raw:
            return json.loads(raw)
        return None
    except Exception as exc:
        logger.warning("Redis get_metadata failed: %s", exc)
        return None


def get_cache_stats() -> dict:
    """
    Return Redis INFO stats for Prometheus / the /metrics endpoint.
    Returns empty dict if Redis is unavailable.
    """
    client = _get_client()
    if client is None:
        return {}

    try:
        info = client.info("stats")
        return {
            "redis_hits": info.get("keyspace_hits", 0),
            "redis_misses": info.get("keyspace_misses", 0),
            "redis_connected": True,
        }
    except Exception:
        return {"redis_connected": False}


def invalidate_session_cache(session_id: str) -> int:
    """
    Delete all cached document hashes for a session.
    Useful if a session's Qdrant collection is reset.
    Returns number of keys deleted.
    """
    client = _get_client()
    if client is None:
        return 0

    try:
        pattern = f"{KEY_PREFIX}:{session_id}:*"
        keys = client.keys(pattern)
        if keys:
            deleted = client.delete(*keys)
            logger.info(
                "Cache invalidated: session=%s | deleted=%d keys", session_id[:8], deleted
            )
            return deleted
        return 0
    except Exception as exc:
        logger.warning("Redis cache invalidation failed: %s", exc)
        return 0
