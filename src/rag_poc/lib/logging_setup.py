"""Process-wide logging and warning configuration for CLI runs."""

import logging
import os
import warnings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Third-party loggers that are noisy at INFO and drown out pipeline progress.
_QUIET_LOGGERS = {
    "httpx": logging.WARNING,
    "httpx2": logging.WARNING,
    "urllib3": logging.WARNING,
    "filelock": logging.WARNING,
    "faiss": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    # Repeats an optional "set HF_TOKEN for higher rate limits" hint on every download.
    "huggingface_hub": logging.ERROR,
}


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once for CLI runs; safe to call repeatedly."""
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT, datefmt=_DATE_FORMAT, force=True)
    for name, quiet_level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(quiet_level)
    # huggingface_hub resets its logger level when first imported (lazily, after this runs)
    # from this variable, and prints server hints through its own handler as well as ours.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    # Windows without Developer Mode can't create symlinks; the Hugging Face cache then copies
    # files instead, which is harmless here, so its multi-line warning on each download is noise.
    warnings.filterwarnings(
        "ignore", message=".*cache-system uses symlinks.*", category=UserWarning
    )
