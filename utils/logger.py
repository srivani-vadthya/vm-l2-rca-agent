import logging
import sys
import json
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in ("ticket_id", "correlation_id", "stage", "elapsed_ms", "extra"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)

        return json.dumps(entry)


def _build_logger(name: str) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False
    return log


logger = _build_logger("L2-RCA")


def log(message: str, *, ticket_id: str = "", correlation_id: str = "",
        stage: str = "", elapsed_ms: float = None, extra: dict = None,
        level: str = "info"):

    fields = {}
    if ticket_id:
        fields["ticket_id"] = ticket_id
    if correlation_id:
        fields["correlation_id"] = correlation_id
    if stage:
        fields["stage"] = stage
    if elapsed_ms is not None:
        fields["elapsed_ms"] = round(elapsed_ms, 3)
    if extra:
        fields["extra"] = extra

    record = logging.LogRecord(
        name="L2-RCA",
        level=getattr(logging, level.upper(), logging.INFO),
        pathname="", lineno=0,
        msg=message, args=(), exc_info=None
    )
    for k, v in fields.items():
        setattr(record, k, v)

    logger.handle(record)
