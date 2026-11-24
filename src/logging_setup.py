"""Logging centralizzato.

Tutti i moduli usano `get_logger(__name__)`. I record finiscono:
- in `logs/app.log` (file rotante, persistente tra le sessioni);
- in un buffer in memoria che la GUI legge per la pagina "Log & Errori".

Gli errori con traceback sono anche salvati in un registro dedicato (`ERRORS`),
così la GUI può mostrarli con un identificativo e il dettaglio completo.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR

LOGGER_ROOT = "ccl"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_MAX_RECORDS = 2000
_MAX_ERRORS = 200


@dataclass
class LogEntry:
    time: datetime
    level: str
    logger: str
    message: str
    traceback: str | None = None


@dataclass
class ErrorEntry:
    id: str
    time: datetime
    step: str
    error_type: str
    message: str
    traceback: str
    context: dict = field(default_factory=dict)


class _MemoryHandler(logging.Handler):
    def __init__(self, buffer: deque):
        super().__init__()
        self.buffer = buffer
        self._lock_buf = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        tb = None
        if record.exc_info:
            tb = "".join(traceback.format_exception(*record.exc_info))
        entry = LogEntry(
            time=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            traceback=tb,
        )
        with self._lock_buf:
            self.buffer.append(entry)


LOG_BUFFER: deque[LogEntry] = deque(maxlen=_MAX_RECORDS)
ERRORS: deque[ErrorEntry] = deque(maxlen=_MAX_ERRORS)
_error_total = 0  # contatore monotono (la deque è limitata: la sua lunghezza smette di crescere)
_error_lock = threading.Lock()


def errors_total() -> int:
    return _error_total


_configured = False
_config_lock = threading.Lock()


def setup_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call at every Streamlit rerun."""
    global _configured
    with _config_lock:
        if _configured:
            return
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger(LOGGER_ROOT)
        root.setLevel(logging.DEBUG)
        root.propagate = False

        file_handler = RotatingFileHandler(
            LOGS_DIR / "app.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        file_handler.setLevel(logging.DEBUG)

        mem_handler = _MemoryHandler(LOG_BUFFER)
        mem_handler.setLevel(level)

        root.addHandler(file_handler)
        root.addHandler(mem_handler)

        # Librerie rumorose: solo warning in su.
        for noisy in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{LOGGER_ROOT}.{short}")


def set_level(level: int) -> None:
    for handler in logging.getLogger(LOGGER_ROOT).handlers:
        if isinstance(handler, _MemoryHandler):
            handler.setLevel(level)


def record_error(step: str, exc: BaseException, context: dict | None = None) -> ErrorEntry:
    """Log an exception and keep it in the error registry; returns the entry (with a short id)."""
    entry = ErrorEntry(
        id=uuid.uuid4().hex[:8],
        time=datetime.now(),
        step=step,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        context=context or {},
    )
    global _error_total
    with _error_lock:
        ERRORS.append(entry)
        _error_total += 1
    get_logger("errors").error(
        "[%s] Errore in '%s': %s: %s", entry.id, step, entry.error_type, entry.message,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return entry


def clear_memory() -> None:
    LOG_BUFFER.clear()
    ERRORS.clear()
