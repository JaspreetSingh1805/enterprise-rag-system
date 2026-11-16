import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# Context variable holding the correlation / trace ID for the current async task / thread
trace_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


def get_current_trace_id() -> str:
    """Return the current trace ID or an empty string if unset."""
    return trace_id_ctx.get() or "trace-unset"


def set_current_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context."""
    trace_id_ctx.set(trace_id)


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter injecting timestamp, level, trace_id, and log message.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_current_trace_id(),
        }

        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class TextTraceFormatter(logging.Formatter):
    """
    Human-readable formatter with trace_id for development/terminal debugging.
    """

    def format(self, record: logging.LogRecord) -> str:
        tid = get_current_trace_id()
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{time_str}] [{record.levelname:<8}] [trace_id={tid}] {record.name}: "
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return prefix + msg


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with trace ID formatting."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextTraceFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
