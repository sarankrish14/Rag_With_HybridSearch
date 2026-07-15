"""Logging setup for the semantic chunker."""

import logging
import sys

from semantic_chunker.settings import LOG_FILE


def setup_logging() -> logging.Logger:
  """Create a logger that writes to terminal and chunker.log."""
  logger = logging.getLogger("semantic_chunker")
  logger.setLevel(logging.DEBUG)

  fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
  fh.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)

  fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
  )
  fh.setFormatter(fmt)
  ch.setFormatter(fmt)

  logger.addHandler(fh)
  logger.addHandler(ch)

  return logger


logger = setup_logging()
