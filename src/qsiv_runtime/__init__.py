"""Local QSIV decision model: shared GPU service, stdlib-only client, setup and doctor."""

from __future__ import annotations

from .client import QSIVClient
from .doctor import doctor
from .errors import Cause, QSIVDiagnostic, QSIVUnavailable
from .setup import setup

__version__ = "0.1.0"

__all__ = [
    "Cause",
    "QSIVClient",
    "QSIVDiagnostic",
    "QSIVUnavailable",
    "__version__",
    "doctor",
    "setup",
]
