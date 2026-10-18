
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)



EMBEDDING_VERSION = "text-embedding-3-small-v1"
KEY_PREFIX = "verirag:doc"
_TTL_DAYS = int(os.getenv("REDIS_DOC_TTL_DAYS", "30"))
TTL_SECONDS = _TTL_DAYS * 24 * 3600



_client = None
_client_initialized = False


def _get_client():
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
        import redis  

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


def compute_document_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _make_key(session_id: str, doc_hash: str) -> str:
    return f"{KEY_PREFIX}:{EMBEDDING_VERSION}:{session_id}:{doc_hash}"


def is_document_indexed(doc_hash: str, session_id: str) -> bool:
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

    client = _get_client()
    if client is None:
        return 0

    try:
        pattern = f"{KEY_PREFIX}:{EMBEDDING_VERSION}:{session_id}:*"
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
