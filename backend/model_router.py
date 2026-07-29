""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


_CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))
_CB_RECOVERY_TIMEOUT  = float(os.getenv("CB_RECOVERY_TIMEOUT", "60"))
_MAX_RETRIES          = int(os.getenv("LLM_MAX_RETRIES", "2"))
_BASE_DELAY           = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))

_TRANSIENT_KEYWORDS = (
    "timeout", "timed out", "rate limit", "429", "503", "502",
    "connection", "overloaded", "server error", "try again",
)


class _CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Per-provider circuit breaker.
    Thread-safety: not locked — acceptable for our async-within-sync model
    since LangGraph nodes run sequentially within a thread.
    """
    name: str
    failure_threshold: int   = _CB_FAILURE_THRESHOLD
    recovery_timeout:  float = _CB_RECOVERY_TIMEOUT

    _state:             _CircuitState = field(default=_CircuitState.CLOSED, init=False, repr=False)
    _failure_count:     int           = field(default=0,   init=False, repr=False)
    _last_failure_time: float         = field(default=0.0, init=False, repr=False)


    @property
    def state_name(self) -> str:
        return self._state.value

    def is_open(self) -> bool:
        """
        Returns True if requests should be blocked for this provider.
        Automatically transitions OPEN → HALF_OPEN after recovery_timeout.
        """
        if self._state == _CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "Circuit[%s] OPEN → HALF_OPEN (%.0fs elapsed, probing)",
                    self.name, elapsed,
                )
                self._state = _CircuitState.HALF_OPEN
                return False   # allow the probe request
            return True        # still open, block
        return False


    def record_success(self) -> None:
        if self._state == _CircuitState.HALF_OPEN:
            logger.info("Circuit[%s] HALF_OPEN → CLOSED (probe succeeded)", self.name)
            self._state = _CircuitState.CLOSED
            self._failure_count = 0
        elif self._state == _CircuitState.CLOSED:
            
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            if self._state != _CircuitState.OPEN:
                logger.warning(
                    "Circuit[%s] → OPEN after %d failures",
                    self.name, self._failure_count,
                )
            self._state = _CircuitState.OPEN

        elif self._state == _CircuitState.HALF_OPEN:
            logger.warning("Circuit[%s] HALF_OPEN → OPEN (probe failed)", self.name)
            self._state = _CircuitState.OPEN



@dataclass
class _Provider:
    name: str
    llm: Any                        # BaseChatModel
    circuit_breaker: CircuitBreaker



def _build_providers() -> list[_Provider]:
    """
    Build the ordered list of providers from available env vars.
    Providers are tried in order — first available is primary.
    """
    providers: list[_Provider] = []

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            providers.append(_Provider(
                name="openai-gpt-4o-mini",
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=30,streaming=True),
                circuit_breaker=CircuitBreaker(name="openai"),
            ))
            logger.info("Model router: registered openai-gpt-4o-mini (primary)")
        except Exception as exc:
            logger.warning("OpenAI provider init failed: %s", exc)

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
            providers.append(_Provider(
                name="claude-haiku",
                llm=ChatAnthropic(
                    model="claude-haiku-4-5-20251001",
                    temperature=0,
                    timeout=30,
                ),
                circuit_breaker=CircuitBreaker(name="anthropic"),
            ))
            logger.info("Model router: registered claude-haiku (fallback-1)")
        except ImportError:
            logger.info("langchain-anthropic not installed — Claude fallback unavailable")
        except Exception as exc:
            logger.warning("Anthropic provider init failed: %s", exc)

    # ── Fallback 2: Google Gemini Flash 
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
            providers.append(_Provider(
                name="gemini-flash",
                llm=ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0,
                    timeout=30,
                ),
                circuit_breaker=CircuitBreaker(name="google"),
            ))
            logger.info("Model router: registered gemini-flash (fallback-2)")
        except ImportError:
            logger.info("langchain-google-genai not installed — Gemini fallback unavailable")
        except Exception as exc:
            logger.warning("Google provider init failed: %s", exc)

    if not providers:
        raise RuntimeError(
            "No LLM providers configured. "
            "Set at least OPENAI_API_KEY to enable GPT-4o-mini."
        )

    return providers


_providers: Optional[list[_Provider]] = None


def _get_providers() -> list[_Provider]:
    global _providers
    if _providers is None:
        _providers = _build_providers()
    return _providers


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying (rate limits, timeouts, 5xx)."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def _invoke_with_retry(bound_llm: Any, messages: list, max_retries: int) -> Any:
    """
    Invoke bound_llm with exponential backoff retry on transient errors.
    Non-transient errors (auth failures, bad requests) fail immediately.
    """
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries + 1):
        try:
            return bound_llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == max_retries:
                raise
            delay = _BASE_DELAY * (2 ** attempt)   # 1s, 2s, 4s ...
            logger.warning(
                "LLM transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries + 1, exc, delay,
            )
            time.sleep(delay)

    raise last_exc


def invoke_with_fallback(
    messages: list,
    *,
    structured_output_schema=None,
    tools: Optional[list] = None,
    parallel_tool_calls: bool = False,
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """
    Invoke the LLM through the provider fallback chain.

    Tries each provider in order. Skips providers with open circuit breakers.
    Within each provider, retries transient errors with exponential backoff.

    Args:
        messages:                 LangChain message list or list of dicts.
        structured_output_schema: Pydantic model → uses with_structured_output().
        tools:                    Tool list → uses bind_tools().
        parallel_tool_calls:      Passed to bind_tools(). Default False.
        max_retries:              Transient retry attempts per provider.

    Returns:
        LLM response (same interface as llm.invoke()).

    Raises:
        RuntimeError if ALL providers fail or are circuit-open.
    """
    providers = _get_providers()
    last_exc: Exception = RuntimeError("All providers tried")
    providers_tried: list[str] = []

    for provider in providers:
        cb = provider.circuit_breaker
        name = provider.name

        if cb.is_open():
            logger.debug("Circuit OPEN for %s — skipping", name)
            continue

        # Build bound LLM — structured output or tools or plain
        bound = provider.llm
        if structured_output_schema is not None:
            bound = provider.llm.with_structured_output(structured_output_schema)
        elif tools:
            bound = provider.llm.bind_tools(
                tools, parallel_tool_calls=parallel_tool_calls
            )

        providers_tried.append(name)

        try:
            result = _invoke_with_retry(bound, messages, max_retries)
            cb.record_success()

            if name != providers[0].name:
                logger.info(
                    "MODEL ROUTER FALLBACK: request served by '%s' (primary='%s')",
                    name, providers[0].name,
                )

            return result

        except Exception as exc:
            cb.record_failure()
            last_exc = exc
            logger.warning(
                "Provider '%s' failed (circuit=%s): %s",
                name, cb.state_name, exc,
            )

    raise RuntimeError(
        f"All LLM providers failed after trying: {providers_tried}. "
        f"Last error: {last_exc}"
    ) from last_exc


def get_primary_llm() -> Any:
    """Return the primary provider's LLM for direct use (e.g. tool binding singletons)."""
    return _get_providers()[0].llm


def get_circuit_breaker_status() -> list[dict]:
    """
    Return circuit breaker states for all registered providers.
    Used by /health endpoint and Prometheus metrics.
    """
    return [
        {
            "provider": p.name,
            "circuit_state": p.circuit_breaker.state_name,
            "failure_count": p.circuit_breaker._failure_count,
        }
        for p in _get_providers()
    ]
