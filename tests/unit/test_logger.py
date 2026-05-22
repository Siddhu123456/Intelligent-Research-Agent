import logging
from logging import Logger


logging.getLogger(
    "sentence_transformers",
).setLevel(logging.ERROR)

logging.getLogger(
    "httpx",
).setLevel(logging.WARNING)


def setup_logger(
    name: str,
) -> Logger:
    """Create and configure logger."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    return logging.getLogger(
        name
    )
    
import logging
from logging import Logger


logging.getLogger(
    "sentence_transformers",
).setLevel(logging.ERROR)

logging.getLogger(
    "httpx",
).setLevel(logging.WARNING)


def setup_logger(
    name: str,
) -> Logger:
    """Create and configure logger."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    return logging.getLogger(
        name
    )