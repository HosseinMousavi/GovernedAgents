"""One place to configure logging. The redaction processor is first in the chain, so
nothing downstream (stdout, CloudWatch, OTel log bridge) ever sees raw PII."""

from __future__ import annotations

import logging

import structlog

from store_support.governance.redaction import redact_event


def configure_logging(*, level: int = logging.INFO, json: bool = True) -> None:
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            redact_event,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),  # resolves sys.stdout per logger, not at config time
        cache_logger_on_first_use=False,
    )
