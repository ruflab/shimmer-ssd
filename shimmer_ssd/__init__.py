import logging
import os
from pathlib import Path

from .version import __version__

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEBUG_MODE = bool(int(os.getenv("DEBUG", "0")))
LOGGING_LEVEL = logging.INFO
LOGGER = logging.getLogger("shimmer_ssd")

handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
LOGGER.setLevel(LOGGING_LEVEL)
LOGGER.addHandler(handler)

__all__ = ["__version__", "PROJECT_DIR", "DEBUG_MODE", "LOGGING_LEVEL", "LOGGER"]
