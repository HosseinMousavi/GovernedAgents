"""PII redaction for logs and traces.

Uses Microsoft Presidio when the `pii` extra is installed (NER-backed: names, locations,
etc.). Falls back to a regex engine for email / phone / card / SSN so the guarantee holds
in minimal environments (CI unit jobs, slim containers).

Both engines return the same marker style: ``<EMAIL_ADDRESS>``.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from functools import lru_cache
from typing import Any

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL_ADDRESS", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE_NUMBER", re.compile(r"(?:\+?1[ -.]?)?(?:\(\d{3}\)|\d{3})[ -.]?\d{3}[ -.]?\d{4}\b")),
]


def _regex_redact(text: str) -> str:
    for label, pat in _PATTERNS:
        text = pat.sub(f"<{label}>", text)
    return text


@lru_cache(maxsize=1)
def _presidio() -> tuple[Any, Any] | None:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return None
    try:
        return AnalyzerEngine(), AnonymizerEngine()
    except Exception:  # noqa: BLE001 - missing spaCy model etc.; fall back silently
        return None


# Entity types we treat as PII. Deliberately excludes DATE_TIME, URL, NRP and the like:
# Presidio's NER tags order numbers as dates and SKUs as people (seen in the field).
PII_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_BANK_NUMBER",
)
# Business identifiers are never PII, whatever the NER model thinks.
_BUSINESS_ID = re.compile(r"^(?:(?:SKU|TCK|APT|RFD|REQ)-[A-Z0-9]+|\d{1,8})$", re.IGNORECASE)
_MIN_SCORE = 0.4


def redact_text(text: str) -> str:
    engines = _presidio()
    if engines is not None:
        analyzer, anonymizer = engines
        results = [
            r
            for r in analyzer.analyze(text=text, language="en", entities=list(PII_ENTITIES), score_threshold=_MIN_SCORE)
            if not _BUSINESS_ID.match(text[r.start : r.end].strip())
        ]
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    # Regex pass always runs: cheap, and covers formats NER occasionally misses.
    return _regex_redact(text)


def _walk(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, MutableMapping):
        return {k: _walk(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_walk(v) for v in value)
    return value


def redact_event(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor: redact every string anywhere in the event."""
    return _walk(event_dict)  # type: ignore[no-any-return]
