
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



# Injection patterns — Layer 1


_INJECTION_PATTERNS: list[str] = [
    # ignore-previous-instructions family
    r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above|preceding)\s+instructions",
    r"forget\s+(everything|all)\s+(you|i|we)\s+(were|have|told)",
    r"override\s+(your\s+)?(previous\s+)?instructions",
    # System prompt exfiltration
    r"(print|repeat|output|reveal|show|display|tell me)\s+(your\s+)?(system\s+prompt|instructions|rules|directives)",
    r"what\s+(are|were)\s+your\s+(system\s+)?(instructions|rules|prompt)",
    r"(leak|expose|dump)\s+(your\s+)?(system\s+prompt|context|instructions)",
    # Role override / jailbreak
    r"you\s+(are|will\s+be|must\s+act\s+as)\s+(now\s+)?(a|an|the)?\s*(evil|unrestricted|jailbroken|uncensored)",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(a\s+)?(different|another|new)\s+(AI|model|assistant|chatbot)",
    r"pretend\s+(that\s+)?(you\s+)?(are|have\s+no)\s+(restrictions|rules|guidelines|filter)",
    r"\bDAN\b",          # Do Anything Now
    r"jailbreak",
    r"developer\s+mode",
    r"god\s+mode",
    r"prison\s+break",
    r"do\s+anything\s+now",
    # Prompt delimiter injection
    r"<\s*/?system\s*>",
    r"\[INST\]|\[/INST\]",
    r"###\s*System\s*:",
    r"<\s*human\s*>.*ignore",
    r"SYSTEM\s*:\s*you\s+are\s+now",
    # Encoded attacks (basic base64 keyword detection)
    r"aWdub3Jl",     # base64("ignore")
    r"aW5zdHJ1Y3Rpb25z",  # base64("instructions")
]

_COMPILED_INJECTION = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in _INJECTION_PATTERNS
]


# Hidden instruction patterns in retrieved web content — Layer 2


_HIDDEN_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous",
    r"you\s+must\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you\s+are",
    r"assistant\s*:\s*i\s+will\s+(now\s+)?",
    r"\[hidden\s+prompt\]",
    r"<!--.*?(inject|override|ignore).*?-->",
    r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",  # inline scripts
    r"disregard\s+the\s+(above|previous)",
    r"from\s+now\s+on\s+(you|ignore|forget)",
]

_COMPILED_HIDDEN = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in _HIDDEN_PATTERNS
]


# PII regex fallback patterns — Layer 3


_PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_US":    re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN":         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IP_ADDRESS":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "AADHAAR":     re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),  # Indian ID
}



@dataclass
class GuardrailResult:
    safe: bool
    reason: str = ""
    flagged_pattern: str = ""


@dataclass
class SanitizedContext:
    content: str
    redacted_count: int = 0
    was_clean: bool = True


@dataclass
class ValidationResult:
    safe: bool                              # False = PII was found and redacted
    pii_types: list[str] = field(default_factory=list)
    redacted_output: str = ""



# Layer 1 — Input Guardrail


def check_input(user_input: str) -> GuardrailResult:
    """
    Fast, rule-based prompt injection and jailbreak detection.
    No LLM call. Runs in microseconds.

    Returns GuardrailResult(safe=False) on first match — caller should
    reject the request with HTTP 400 immediately.

    Policy: FAIL CLOSED — when in doubt, block.
    """
    if not user_input or not user_input.strip():
        return GuardrailResult(safe=False, reason="Empty input.", flagged_pattern="empty_input")

    if len(user_input) > 10_000:
        return GuardrailResult(
            safe=False,
            reason="Input exceeds 10,000 character limit.",
            flagged_pattern="max_length_exceeded",
        )

    for pattern in _COMPILED_INJECTION:
        if pattern.search(user_input):
            logger.warning(
                "INPUT GUARDRAIL TRIGGERED | pattern='%s...' | input_preview='%s...'",
                pattern.pattern[:50],
                user_input[:80].replace("\n", " "),
            )
            return GuardrailResult(
                safe=False,
                reason=(
                    "Your input contains patterns associated with prompt injection "
                    "or jailbreak attempts and cannot be processed."
                ),
                flagged_pattern=pattern.pattern[:60],
            )

    return GuardrailResult(safe=True)


# Layer 2 — Context Sanitizer


def sanitize_context(raw_text: str) -> SanitizedContext:
    """
    Remove hidden instructions injected into retrieved web content.
    This guards against indirect / retrieval-poisoning attacks where
    a malicious web page embeds instructions directed at the LLM.

    Policy: FAIL OPEN — if this errors, log and return raw text.
    """
    if not raw_text:
        return SanitizedContext(content=raw_text, redacted_count=0, was_clean=True)

    try:
        sanitized = raw_text
        total_redacted = 0

        for pattern in _COMPILED_HIDDEN:
            matches = pattern.findall(sanitized)
            if matches:
                total_redacted += len(matches)
                sanitized = pattern.sub("[CONTENT REDACTED — HIDDEN INSTRUCTION]", sanitized)

        if total_redacted > 0:
            logger.warning(
                "CONTEXT SANITIZER: removed %d hidden instruction(s) from retrieved content",
                total_redacted,
            )

        return SanitizedContext(
            content=sanitized,
            redacted_count=total_redacted,
            was_clean=(total_redacted == 0),
        )

    except Exception as exc:
        logger.error("Context sanitizer error (FAIL OPEN): %s", exc)
        return SanitizedContext(content=raw_text, redacted_count=0, was_clean=True)


# Layer 3 — Output Validator (PII Redaction)


def _presidio_redact(text: str) -> tuple[str, list[str]]:
    """
    Attempt PII redaction using Microsoft Presidio.
    Raises ImportError if Presidio not installed.
    """
    from presidio_analyzer import AnalyzerEngine  # type: ignore
    from presidio_anonymizer import AnonymizerEngine  # type: ignore

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text, []

    entity_types = sorted({r.entity_type for r in results})
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, entity_types


def _regex_redact(text: str) -> tuple[str, list[str]]:
    """Regex-based PII redaction fallback."""
    found: list[str] = []
    redacted = text
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(redacted):
            found.append(pii_type)
            redacted = pattern.sub(f"[{pii_type} REDACTED]", redacted)
    return redacted, found


def validate_output(llm_output: str) -> ValidationResult:
    """
    Scan LLM-generated output for PII before returning it to the user.

    Tries Presidio first (richer detection — names, orgs, dates, etc.).
    Falls back to regex patterns if Presidio is not installed.

    Policy: FAIL OPEN — if validation errors, return original output.
            A failed PII scan is better than a broken user experience.
    """
    if not llm_output or not llm_output.strip():
        return ValidationResult(safe=True, redacted_output=llm_output)

    try:
        try:
            redacted, pii_types = _presidio_redact(llm_output)
        except ImportError:
            redacted, pii_types = _regex_redact(llm_output)

        if pii_types:
            logger.warning(
                "OUTPUT VALIDATOR: PII found in LLM output — types=%s (redacted)",
                pii_types,
            )
            return ValidationResult(
                safe=False,
                pii_types=pii_types,
                redacted_output=redacted,
            )

        return ValidationResult(safe=True, redacted_output=llm_output)

    except Exception as exc:
        logger.error("Output validator error (FAIL OPEN): %s", exc)
        return ValidationResult(safe=True, redacted_output=llm_output)
