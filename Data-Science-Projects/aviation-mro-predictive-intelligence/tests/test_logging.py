"""Tests for the structured logging service (SRS v1.1 NFR-07).

Covers logger creation, log levels, secret redaction, and no sensitive
information leakage in log output.
"""

from __future__ import annotations

import logging

from aviation_mro.services.logging import (
    _SecretRedactingFilter,
    get_logger,
    redact,
    setup_logging,
)


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_setup_logging_idempotent(self) -> None:
        """Calling setup_logging multiple times does not add duplicate handlers."""
        root = logging.getLogger()
        initial_handler_count = len(root.handlers)
        setup_logging()
        setup_logging()
        assert len(root.handlers) == initial_handler_count

    def test_setup_logging_sets_level(self) -> None:
        """setup_logging sets the root logger level."""
        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        setup_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_redact_masks_secret_values(self) -> None:
        """redact() masks secret values while preserving key names."""
        assert redact("password=secret123") == "password=[REDACTED]"
        assert redact("api_key=abc-def-ghi") == "api_key=[REDACTED]"
        assert redact("token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") == "token=[REDACTED]"
        assert redact("normal message without secrets") == "normal message without secrets"

    def test_redact_case_insensitive(self) -> None:
        """redact() is case-insensitive for secret keywords."""
        assert redact("PASSWORD=hidden") == "PASSWORD=[REDACTED]"
        assert redact("Api_Key=hidden") == "Api_Key=[REDACTED]"
        assert redact("TOKEN=hidden") == "TOKEN=[REDACTED]"

    def test_redact_multiple_secrets(self) -> None:
        """redact() masks multiple secrets in one message."""
        result = redact("user=alice password=pwd123 token=abc token2=xyz")
        assert "pwd123" not in result
        assert "[REDACTED]" in result
        assert "user=alice" in result  # Non-secret key-value preserved


class TestLoggingSecretRedaction:
    """Tests that secrets do not leak into log output."""

    def test_filter_redacts_password_value(self) -> None:
        """The redacting filter masks password values in log records."""
        redacting_filter = _SecretRedactingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Connecting with password=supersecret123 to db",
            args=(),
            exc_info=None,
        )
        redacting_filter.filter(record)
        assert "supersecret123" not in record.getMessage()
        assert "password=[REDACTED]" in record.getMessage()

    def test_filter_redacts_api_key_value(self) -> None:
        """The redacting filter masks API key values in log records."""
        redacting_filter = _SecretRedactingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Calling API with api_key=sk-1234567890",
            args=(),
            exc_info=None,
        )
        redacting_filter.filter(record)
        assert "sk-1234567890" not in record.getMessage()
        assert "api_key=[REDACTED]" in record.getMessage()

    def test_logger_normal_message_passthrough(self, caplog) -> None:
        """Normal (non-secret) log messages are not modified."""
        setup_logging(level="INFO")
        logger = get_logger("test.normal")
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            logger.info("Processing engine 42 RUL prediction")
        captured = caplog.text
        assert "engine 42" in captured
        assert "RUL prediction" in captured


class TestLoggingLevels:
    """Tests for log level filtering."""

    def test_debug_message_hidden_at_info(self, caplog) -> None:
        """DEBUG messages are hidden when level is INFO."""
        setup_logging(level="INFO")
        logger = get_logger("test.levels.debug")
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            logger.debug("this is a debug message")
            logger.info("this is an info message")
        captured = caplog.text
        assert "this is a debug message" not in captured
        assert "this is an info message" in captured

    def test_error_message_always_visible(self, caplog) -> None:
        """ERROR messages are visible at any level."""
        setup_logging(level="WARNING")
        logger = get_logger("test.levels.error")
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            logger.error("this is an error message")
        captured = caplog.text
        assert "this is an error message" in captured
