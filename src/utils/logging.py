"""
Logger standard uniforme per tutti gli script.
"""
import logging, sys

def get_logger(name: str = "nanosocrates"):
    """Return a shared logger configured with consistent formatting."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # Configure the logger only once so repeated calls remain inexpensive.
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
