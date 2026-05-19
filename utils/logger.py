import logging
from logging import Logger

import logging

logging.getLogger(
    "sentence_transformers",
).setLevel(logging.ERROR)

logging.getLogger(
    "httpx",
).setLevel(logging.WARNING)


def setup_logger(name: str) -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    return logging.getLogger(name)