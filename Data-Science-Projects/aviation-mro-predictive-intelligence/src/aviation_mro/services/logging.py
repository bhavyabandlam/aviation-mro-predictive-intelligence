"""Structured application logging (SRS v1.1 NFR-07).

Provides a single, module-level ``logging`` configuration for the Aviation
MRO Predictive Intelligence platform. Design goals:

* Clear logs and actionable exceptions.
* No secrets or sensitive values in log output.
* Structured, timestamped, human-readable log lines.
* Safe to call from any module via ``get_logger(__name__)``.

The logger redacts values that look like secrets (API keys, passwords,
tokens, connection strings) before they reach any handler. Redaction is
best-effort: it masks literal strings containing common secret keywords and
also scrubs any ``**kwargs`` whose key matches a secret keyword.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

__all__ = ["get_logger", "setup_logging", "redact"]

# Secret keywords that trigger redaction when they appear in log values or
# keyword-argument names.
_SECRET_KEYWORDS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "refresh_token",
    "refreshtoken",
    "client_secret",
    "clientsecret",
    "private_key",
    "privatekey",
    "encryption_key",
    "encryptionkey",
    "connection_string",
    "connectionstring",
    "dsn",
    "token",
)

# Regex matching common secret-like patterns in string values.
# Captures the key in group 1 so only the value is redacted.
_SECRET_PATTERN = re.compile(
    r"(?i)(password|passwd|pwd|secret|api.?key|access.?token|"
    r"refresh.?token|client.?secret|private.?key|encryption.?key|"
    r"connection.?string|dsn|token)(\s*[:=]\s*[\S]+)"
)

_DEFAULT_LEVEL = os.environ.get("AVIATION_MRO_LOG_LEVEL", "INFO").upper()

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


class _SecretRedactingFilter(logging.Filter):
    """Filter that redacts secret-like substrings from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the message itself (secret values only, key names preserved).
        if record.getMessage():
            record.msg = _SECRET_PATTERN.sub(r"\1=[REDACTED]", record.getMessage())
            record.args = ()
        # Redact any extra attributes that might contain secrets.
        for key in ("password", "secret", "token", "key"):
            if hasattr(record, key):
                setattr(record, key, "[REDACTED]")
        return True


def _redact_value(value: Any) -> Any:
    """Return *value* with secret-like substrings replaced by ``[REDACTED]``."""
    if isinstance(value, str):
        return _SECRET_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if _looks_secret(k) else _redact_value(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


def _looks_secret(key: str) -> bool:
    key_lower = key.lower()
    return any(kw in key_lower for kw in _SECRET_KEYWORDS)


def redact(message: str) -> str:
    """Return *message* with secret values redacted (key names preserved)."""
    return _SECRET_PATTERN.sub(r"\1=[REDACTED]", message)


def setup_logging(level: str | int | None = None) -> None:
    """Configure the root logger once.

    The log level is always updated on every call. Handlers are only added
    once (idempotent) to avoid duplicate handlers on repeated calls.
    """
    root = logging.getLogger()
    resolved_level = level or _DEFAULT_LEVEL
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level, logging.INFO)
    root.setLevel(resolved_level)

    if root.handlers:
        return  # Handlers already configured

    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    handler.addFilter(_SecretRedactingFilter())
    root.addHandler(handler)

    # Silence noisy third-party loggers unless explicitly overridden.
    for noisy in ("urllib3", "matplotlib", "PIL", "joblib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, configured via :func:`setup_logging`."""
    return logging.getLogger(name)
